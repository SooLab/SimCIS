# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import BitMasks

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils import box_ops

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import pickle
import torch.distributed as dist
import copy

@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        output_dir:str,
        current_catagory_ids: list = None,
        task: int = 1,
        psd_label_threshold: float = 0.35,
        psd_overlap_threshold: float = 0.8,
        collect_query_mode: bool = False,
        lib_size: int = 80,
        combine_psdlabel: bool = False,
        vq_number: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        if current_catagory_ids is not None:
            self.current_catagory_ids = torch.tensor(current_catagory_ids)
        
        self.output_dir = output_dir
        
        self.save = {
            'gt': 0,
            'pesudo': 0
        }
        self.task = task
        # print(f"current_catagory_ids: {task}")
        self.psd_overlap_threshold = psd_overlap_threshold
        self.psd_label_threshold = psd_label_threshold
        # self.writer = SummaryWriter(log_dir=f"output/ps/100-10_psd0.8/step{task}")
        self.count = 0
        self.psd_num = torch.zeros(150)

        self.collect_query_mode = collect_query_mode
        if self.task < 10:
            query_root = self.output_dir[:-1] + f"{task-1}"
        else:
            query_root = self.output_dir[:-2] + f"{task-1}"


        with torch.no_grad():
            self.collect = {}
            try:
                if self.task > 1 and not self.collect_query_mode:
                    self.collect = torch.load(f"{query_root}/fake_query.pkl", map_location='cpu')
            except:
                for i in range(10):
                    print(f"************No query found in {query_root}***********************")
        self.lib_size = lib_size
        self.combine_psdlabel = combine_psdlabel
        self.vq_number = vq_number
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        kl_weight = cfg.CONT.KL_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"interm_loss_ce": class_weight, "loss_ce": class_weight, "interm_loss_mask": mask_weight, \
            "loss_mask": mask_weight, "interm_loss_dice": dice_weight, "loss_dice": dice_weight, "interm_loss_bbox": mask_weight, \
                "loss_giou": 2.0, "interm_loss_giou": 2.0,"loss_bbox": mask_weight, "interm_kl_loss" : kl_weight, "kl_loss" : kl_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "points", "kd"] if cfg.CONT.TASK > 1 and cfg.CONT.OLD_MODEL and cfg.CONT.KL_WEIGHT > 0 else ["labels", "masks", "points"]
        # losses = ["labels", "masks"]
        num_tasks = 1 + (cfg.CONT.TOT_CLS - cfg.CONT.BASE_CLS) // cfg.CONT.INC_CLS
        n_cls_in_tasks = [cfg.CONT.BASE_CLS] + [cfg.CONT.INC_CLS] * (num_tasks - 1)
        current_catagory_ids = list(
            range(sum(n_cls_in_tasks[:cfg.CONT.TASK - 1]), sum(n_cls_in_tasks[:cfg.CONT.TASK]))
        )

        criterion = SetCriterion(
            # sem_seg_head.num_classes,
            # sum([cls_embed.out_features for cls_embed in sem_seg_head.predictor.class_embeds]),
            num_classes=150,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            current_catagory_ids=current_catagory_ids,
            vq_number=cfg.CONT.VQ_NUMBER,
            kl_all=cfg.CONT.KL_ALL,
            kd_type=cfg.CONT.KD_TYPE,
            kd_temperature = cfg.CONT.KD_TEMPERATURE,
            kd_temperature2 = cfg.CONT.KD_TEMPERATURE2,
            filter_kd=cfg.CONT.FILTER_KD,
            kd_decoder=cfg.CONT.KD_DECODER,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "current_catagory_ids": current_catagory_ids,
            "task": cfg.CONT.TASK,
            "psd_label_threshold": cfg.CONT.PSD_LABEL_THRESHOLD,
            "psd_overlap_threshold": cfg.CONT.PSD_OVERLAP_THRESHOLD,
            "collect_query_mode": cfg.CONT.COLLECT_QUERY_MODE,
            "output_dir": cfg.OUTPUT_DIR,
            "lib_size" : cfg.CONT.LIB_SIZE,
            "combine_psdlabel": cfg.CONT.COMBINE_PSDLABEL,
            "vq_number": cfg.CONT.VQ_NUMBER,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, old_pred=None, psd_label=False, topk_feats_info=None, old_outputs=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
                'topk_feats_info': {'topk_proposals':topk_proposals, 'med_feats':tgt_undetach, 'class_logits':enc_output_class},
                'med_feats_info': {'flatten_feats':feats.transpose(0, 1), 'feats_logits':enc_outputs_class_unselected },
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        # med_tokens = outputs.pop("med_tokens")
        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            if topk_feats_info is not None:
                distill_positions = topk_feats_info['topk_proposals']
                # distill_positions = self.filter_distillpoints(distill_positions, features, target)
            else:
                distill_positions = None
            if self.task>1 and not self.collect_query_mode and self.vq_number > 0:
                outputs,  _fake_query_labels = self.sem_seg_head(features, distill_positions=distill_positions, query_lib=self.collect)
            else:
                outputs, _fake_query_labels = self.sem_seg_head(features, distill_positions=distill_positions)

            if old_pred is not None:
                psd_targets, old_targets = self.generate_psd_targets(targets, old_pred, gt_instances, False)
                precise_targets, _ = self.generate_psd_targets(targets, old_pred, gt_instances, False)

                # Save the distribution of pseudo labels
                if self.collect_query_mode:
                    for t in psd_targets:
                        for i in t['labels']:
                            self.psd_num[i] += 1
                    self.count += 1

                losses = self.criterion(outputs, targets, psd_targets, old_targets, topk_feats_info, old_outputs, _fake_query_labels)
            else:
                losses = self.criterion(outputs, targets)

            # **************** START store fake query****************
            with torch.no_grad():
                outputs_without_aux = {k: v for k, v in outputs.items() if 'pred' in k}
                complete_psd_targets = []
                if old_pred is not None:
                    for i, (p, t) in enumerate(zip(precise_targets, targets)):
                        complete_psd_targets.append(
                            {
                                'labels': torch.cat([p['labels'], t['labels']]),\
                                'masks': torch.cat([p['masks'], t['masks']]),\
                                'boxes': torch.cat([p['boxes'], t['boxes']])
                            }
                        )
                else:
                    complete_psd_targets = targets
                indices = self.criterion.matcher(outputs_without_aux, complete_psd_targets)
                for index, (indice, target) in enumerate(zip(indices, complete_psd_targets)):
                    labels = target['labels']
                    for q, l in zip(*indice):
                        gt_class = labels[l].item()
                        if gt_class not in self.collect:
                            if dist.is_initialized():
                                world_size = dist.get_world_size()
                            else:
                                world_size = 1 
                            maxlen = self.lib_size //world_size
                            self.collect[gt_class] = deque(maxlen=maxlen)
                        
                        self.collect[gt_class].append(outputs['topk_feats_info']['med_feats'][index][q].tolist())
            # ****************END store fake query****************

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs = self.sem_seg_head(features)
            
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            bbox_pred_results = outputs["pred_boxes"]
            topk_feats_info = outputs["topk_feats_info"]
            old_outputs = outputs
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            # del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result,bbox_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, bbox_pred_results, batched_inputs, images.image_sizes
            ):
                if psd_label:
                    output_psd_label = {}

                    scores, labels = mask_cls_result.sigmoid().max(-1)
                    # n_cls = sum([cls_embed.out_features for cls_embed in self.sem_seg_head.predictor.class_embeds])
                    n_cls = sum([cls for cls in self.sem_seg_head.predictor.n_cls_in_tasks])
                    keep = labels.ne(n_cls) & (scores > self.psd_label_threshold)
                    # keep = labels.ne(n_cls) & (scores > 0.0)

                    # use in PS
                    if not self.combine_psdlabel:
                        T = 0.06 
                        scores, labels = F.softmax(mask_cls_result.sigmoid() / T, dim=-1).max(-1)
                    sort = scores[keep].argsort(descending=True)

                    output_psd_label['labels'] = labels[keep][sort]
                    output_psd_label['masks'] = mask_pred_result[keep][sort].sigmoid()
                    output_psd_label['scores'] = scores[keep][sort]
                    output_psd_label['boxes'] = bbox_pred_result[keep][sort]
                    # output_psd_label['med_feats'] = {}


                    processed_results.append(output_psd_label)
                else:
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r
                    
                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

            if psd_label:
                return processed_results, topk_feats_info, old_outputs
            else:
                return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor) / image_size_xyxy
                }
            )
        return new_targets

    def generate_psd_targets(self, gt_targets, old_preds, gt_instances, strict_mode=False):
        if self.current_catagory_ids is not None:
            # memory_part = [bool(
            #     torch.logical_not(torch.isin(tgt['labels'], self.current_catagory_ids.to(tgt['labels'].device))).sum() != 0
            # ) for tgt in gt_targets]
            memory_part = [bool(
                np.logical_not(np.isin(tgt['labels'].cpu().numpy(), self.current_catagory_ids.cpu().numpy())).sum() != 0
            ) for tgt in gt_targets]
        else:
            memory_part = None

        psd_targets = []
        old_targets = []

        for i, (gt_target, old_pred) in enumerate(zip(gt_targets, old_preds)):
            h, w  = gt_instances[i].image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            psd_target = {}
            old_target = {}

            if strict_mode: #False
                keep = old_pred["scores"] >= 0.4
            else:
                keep = old_pred["scores"] >= 0.0

            old_masks = old_pred["masks"][keep]
            old_labels = old_pred["labels"][keep]
            old_scores = old_pred["scores"][keep]
            old_boxes = old_pred["boxes"][keep]

            # old_masks = old_pred["masks"]
            # old_labels = old_pred["labels"]
            # old_scores = old_pred["scores"]
            # old_boxes = old_pred["boxes"]
            # print(old_labels, len(old_labels.unique()))

            if old_masks.shape[0] == 0:
                psd_target["labels"] = old_labels
                psd_target["masks"] = (old_masks >= 0.5)
                psd_target["boxes"] = (old_boxes)
                psd_targets.append(psd_target)
                old_target["labels"] = old_labels
                old_target["masks"] = (old_masks >= 0.5)
                old_target["boxes"] = (old_boxes)
                old_targets.append(old_target)
                continue

            old_area = (old_masks >= 0.5).sum(dim=(1, 2))
            prob_masks = old_scores.view(-1, 1, 1) * old_masks
            mask_ids = prob_masks.argmax(0)

            non_ol_masks = torch.zeros_like(old_masks).to(torch.bool)
            selected_masks = torch.zeros_like(old_masks).to(torch.bool)
            for k in range(old_labels.shape[0]):
                non_ol_masks[k] = (mask_ids == k) & (old_masks[k] >= 0.5)
                selected_masks[k] = (mask_ids == k)

            # if not memory_part[i]:
            gt_region = torch.clamp(gt_target["masks"].sum(0).int(), min=0, max=1)
            non_ol_masks = torch.logical_and(non_ol_masks, torch.logical_not(gt_region))

            new_area = non_ol_masks.sum(dim=(1, 2))
            argmax_area = selected_masks.sum(dim=(1, 2))
            # keep = (new_area > 0) & (new_area > old_area * self.overlap_threshold)
            if strict_mode: # False
                keep = (new_area > 0) & (new_area > old_area * self.psd_overlap_threshold) & (old_area > argmax_area * self.psd_overlap_threshold)
            else:
                keep = (new_area > 0) & (new_area > old_area * self.psd_overlap_threshold) #& (old_area > argmax_area * self.psd_overlap_threshold)

            psd_target["labels"] = old_labels[keep]
            # psd_target["masks"] = non_ol_masks[keep]
            select_masks = non_ol_masks[keep]
            # mask = BitMasks(
            #     torch.stack([x.clone().contiguous() for x in select_masks ])
            # )
            mask = BitMasks(select_masks.clone().contiguous())

            if self.combine_psdlabel:
                unique_labels = psd_target["labels"].unique()
                fused_psd_target = {"labels": [], "masks": [], "boxes": []}
                fused_psd_target['labels'] = unique_labels
                temp_masks = []
                for label in unique_labels:
                    label_mask = select_masks[psd_target["labels"] == label].sum(dim=0).clamp(min=0, max=1)
                    temp_masks.append(label_mask)
                if len(temp_masks) > 0:
                    temp_masks = torch.stack(temp_masks, dim=0)
                    temp_masks = BitMasks(temp_masks.clone().contiguous())
                    fused_psd_target['masks'] = temp_masks.tensor.to(old_labels.device)
                    fused_psd_target['boxes'] = box_ops.box_xyxy_to_cxcywh(temp_masks.get_bounding_boxes().tensor).to(old_labels.device)/image_size_xyxy
                else:
                    temp_masks = torch.tensor([],device=unique_labels.device) 
                    fused_psd_target['masks'] = temp_masks
                    fused_psd_target['boxes'] = temp_masks # all tensor([])
                psd_targets.append(fused_psd_target)
                old_targets.append(copy.deepcopy(fused_psd_target))
                # for dd, mask in enumerate(fused_psd_target['masks']):
                #     import cv2
                #     cv2.imwrite(f"./temp/{i}_{k}_{fused_psd_target['labels'][dd]}.png", mask.cpu().numpy()*255)
                    
            else:
                psd_target["masks"] = mask.tensor.to(old_labels.device)
                psd_target["boxes"] = box_ops.box_xyxy_to_cxcywh(mask.get_bounding_boxes().tensor).to(old_labels.device)/image_size_xyxy
                psd_targets.append(psd_target)

                old_target["labels"] = old_labels[keep]
                old_target["masks"] = mask.tensor.to(old_labels.device)
                old_target["boxes"] = box_ops.box_xyxy_to_cxcywh(mask.get_bounding_boxes().tensor).to(old_labels.device)/image_size_xyxy
                # old_target["scores"] = old_scores[keep]
                # old_target["masks"] = old_masks[keep]
                # old_target["boxes"] = old_boxes[keep]
                old_targets.append(old_target)

        return psd_targets, old_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = mask_cls.sigmoid()
        scores, labels = mask_cls.max(-1)
        keep = labels.ne(sum([cls for cls in self.sem_seg_head.predictor.n_cls_in_tasks])) & (scores >= 0.4)
        mask_pred = mask_pred.sigmoid()

        # OPEN FOR PANOPTIC SEGMENTATION!!!
        T = 0.06
        scores, labels = F.softmax(mask_cls/T, dim=-1).max(-1)

        h, w = mask_pred.shape[-2:]
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]  # sigmoid done up.

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        semseg = torch.ones((h, w), dtype=torch.long, device=cur_masks.device)*self.sem_seg_head.num_classes

        if cur_masks.shape[0] == 0:
            # We didn't detect an mask :(
            # semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            semseg = F.one_hot(semseg, self.sem_seg_head.num_classes+1).float().permute(2, 0, 1)
        else:
            # learn from https://github.com/clovaai/ECLIPSE/blob/main/mask2former/maskformer_model.py#L389 

            cur_mask_ids = cur_prob_masks.argmax(0)
            r = 0.5
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                mask_area = (cur_mask_ids == k).sum().item()
                # mask_area = ((cur_mask_ids == k)& (cur_masks[k] >= r)).sum().item()
                original_area = (cur_masks[k] >= r).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= r)
                #fixme mask_area should be computed differently !
                # mask_area = mask.sum().item()

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    # print(mask_area, original_area, mask.sum().item())
                    if mask_area / original_area < self.overlap_threshold or original_area / mask_area < self.overlap_threshold: 
                        continue
                    semseg[mask] = pred_class 
            semseg = F.one_hot(semseg, self.sem_seg_head.num_classes+1).float().permute(2, 0, 1)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        # mask_pred = mask_pred.sigmoid()
            
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.   

        # manu_cls = mask_cls.sigmoid()
        # manu_cls[..., -10:] *= 1.5
        # scores, labels = manu_cls.max(-1)
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)

        # OPEN FOR PANOPTIC SEGMENTATION
        T = 0.06 
        scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        scores = mask_cls.sigmoid()
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
