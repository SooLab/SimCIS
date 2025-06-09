# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from mask2former.utils import box_ops
import time
import copy

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, mask=None):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    # mask = torch.ones_like(inputs)
    # mask[..., :100] = 0.0
    # mask += targets
    # old_mask = (mask==0.0)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    # loss[old_mask] = 0.0
    if mask is not None:
        loss = loss * mask

    return loss.mean(1).sum() / num_boxes

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, current_catagory_ids=None, 
                    vq_number=3, kl_all=True, kd_type='kl', kd_temperature=1.0, kd_temperature2=0.1, filter_kd=False, kd_decoder=True):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        # self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25
        self.current_catagory_ids = torch.tensor(current_catagory_ids)
        self.vq_number = vq_number
        self.kl_all = kl_all
        self.kd_type = kd_type
        self.kd_temperature = kd_temperature
        self.kd_temperature2 = kd_temperature2
        self.filter_kd = filter_kd
        self.kd_deocder = kd_decoder

    def loss_labels_ce(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        assert 'pred_masks' in outputs
        src_masks = outputs["pred_masks"]
        # 拿到每个batch的gt mask
        masks = [t["masks"] for t in targets]

        src_masks = F.interpolate(
            src_masks,
            size=(masks[0].shape[-2], masks[0].shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        # TODO use valid to mask invalid areas due to padding in loss
        mask_thresh_hold = 0.0
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_logits)
        target_masks = target_masks.sum(1) != 0
        b, q, w, h = src_masks.shape
        omit_query = torch.ones((b, q), dtype=torch.bool, device=src_logits.device)
        for i, (src_mask, src_target) in enumerate(zip(src_masks, target_masks)):
            for j, mask in enumerate(src_mask):
                assert mask.shape == src_target.shape, f"{mask.shape}{src_target.shape}"

                # print(f"{((mask >= mask_thresh_hold) & src_target).sum().item()} VS {(mask >= mask_thresh_hold).sum().item()}")
                # time.sleep(0.1)
                
                if ((mask >= mask_thresh_hold) & src_target).sum().item() <= (mask >= mask_thresh_hold).sum().item() * 0.0:
                    omit_query[i][j]=0
        # print(f"omit_query: {(omit_query==0).sum()}")
        query_mask = omit_query.unsqueeze(-1).repeat(1, 1, src_logits.shape[-1])

        """We don't use query mask"""
        query_mask = None
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        try: 
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        except:
            raise ValueError(f"out of boundry {target_classes_onehot.shape} but got {target_classes}")

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2, mask=query_mask) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_bboxes_panoptic(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        # print("***********debug***********")
        # print(idx)
        # print(targets[0]['boxes'].shape, outputs['pred_boxes'].shape)
        # for t, (_, i) in zip(targets, indices):
        #     print(i)
        #     print(t['boxes'].shape)

        # print("***********debug***********")

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # we only need cxcy
        # target_boxes = target_boxes[:, :2]
        
        # For ADE200k
        # stuff_idx = np.array([0, 1, 2, 3, 4, 5, 6, 9, 11, 13, 16, 17, 21, 25, 26, 28, 29, 34, 40, 46, 48, 51, 52, \
        #     54, 59, 60, 61, 63, 68, 77, 79, 84, 91, 94, 96, 99, 100, 101, 105, 106, 109, 113, 114, 117, 122, 128, 131, 140, 141, 145])
        stuff_idx = np.array([])

        isthing = ~np.isin(target_labels.cpu().numpy(), stuff_idx)
        target_boxes=target_boxes[isthing]
        src_boxes=src_boxes[isthing]
        # print('modify')

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        try: 
            src_masks = src_masks[src_idx]
            # src_masks = src_masks[(0,0)]
        except:
            raise ValueError(f"out of boundry {src_masks.shape} but got {src_idx}")
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses
    
    def loss_knowledge_distillation(self, outputs, targets, t, num_masks):
        distill_logits = outputs['pred_logits']
        old_logits = targets['pred_logits']
        if targets is not None and self.kd_type == 'kl': # We use kl in this work(SimCIS)
            if not self.kl_all:
                old_class_num = min(self.current_catagory_ids)
                # mask = torch.max(old_logits[...,:old_class_num], dim=-1)[0] > torch.sum(old_logits[...,old_class_num:], dim=-1)
                # distill_logits = distill_logits[mask]
                # old_logits = old_logits[mask]
                distill_logits = distill_logits[...,:old_class_num]
                old_logits = old_logits[...,:old_class_num]
            # select = old_logits.max(-1)[0] > 0.4
            # distill_logits = distill_logits[select]
            # old_logits = old_logits[select]           
            T = t
            distill_probs = F.log_softmax(distill_logits/T, dim=-1)
            old_probs = F.softmax(old_logits/T, dim=-1)
            bs = old_probs.shape[0]

            # Select with old_probs_entropy 
            # if self.filter_kd:
            #     entropy = -torch.sum(old_probs * torch.log(old_probs), dim=-1)
            #     # median_entropy = torch.median(entropy, dim=1, keepdim=True)[0]
            #     median_entropy = torch.quantile(entropy, q=0.5, dim=1, keepdim=True)
            #     # print("entropy:",median_entropy)
            #     select = entropy < median_entropy
            #     # print( -torch.sum(old_probs * torch.log(old_probs), dim=-1))
            #     # print("select: ",select.sum())
            #     distill_probs = distill_probs[select].view(bs,int(select.sum()/bs),-1)
            #     old_probs = old_probs[select].view(bs,int(select.sum()/bs),-1)
                # print(f"distill_probs: {distill_probs.shape}, old_probs: {old_probs.shape}")
            # if t<1.0:
            #     show_dis = F.softmax(distill_logits, dim=-1)[0,:10]
            #     show_old = F.softmax(old_logits, dim=-1)[0,:10]
            #     print(show_dis.shape)
            #     print('dis:',torch.topk(show_dis, 5, dim=-1))            
            #     print('dis_entropy:',-torch.sum(show_dis * torch.log(show_dis), dim=-1))            
            #     print('old:',torch.topk(show_old, 5, dim=-1))            
            #     print('old_entropy:',-torch.sum(show_old * torch.log(show_old), dim=-1))            
            kl_loss = F.kl_div(distill_probs, old_probs, reduction='batchmean') #* (T**2)


            # bs = distill_logits.shape[0]
            # for i in range(bs):
            #     max_scores, _ = distill_logits[i].max(-1)
            #     max_thresh = max_scores.mean() + 2 * max_scores.std()
            #     mask = max_scores.sigmoid() < 0.4
            #     # print(max_scores.sigmoid().sort())
            #     kl_loss[i][mask] = 0.0
            # kl_loss = kl_loss.sum(-1).mean()
            
            return {'kl_loss': kl_loss}
        elif self.kd_type == 'l2':
            loss = self.L2_distillation_loss(distill_logits, old_logits)
            return {'kl_loss': loss}
        elif self.kd_type == 'ukd':
            targets = old_logits.transpose(1, 2)
            targets = targets * 1
            inputs = distill_logits.transpose(1, 2)

            den = torch.logsumexp(inputs, dim=-1)  # B, Q
            outputs_no_bgk = inputs - den.unsqueeze(dim=1)  # B, OLD_CL, Q
            # outputs_bkg = torch.logsumexp(inputs[:, targets.shape[1]-1:], dim=1) - den  # B, Q
            labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, Q
            labels_soft = torch.log_softmax(targets, dim=1)

            loss = (labels * (labels_soft - outputs_no_bgk)).sum(dim=1)  # B, Q
            # Re-weight no-cls queries as in classificaton
            return loss.mean()
        else:
            raise ValueError("topk_feats_info is None, please set it to compute knowledge distillation loss")

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'points': self.loss_bboxes_panoptic,
            'kd': self.loss_knowledge_distillation,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, psd_targets=None, old_targets=None, topk_feats_info=None, old_outputs=None, fake_query_labels=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'pred' in k}

        if self.current_catagory_ids is not None:
            # print("current_catagory_ids", self.current_catagory_ids)
            # memory_part = [bool(
            #     torch.logical_not(torch.isin(tgt['labels'], self.current_catagory_ids.to(tgt['labels'].device))).sum() != 0
            # ) for tgt in targets]
            memory_part = [bool(
                np.logical_not(np.isin(tgt['labels'].cpu().numpy(), self.current_catagory_ids.cpu().numpy())).sum() != 0
            ) for tgt in targets]
        else:
            memory_part = None

        if psd_targets is None or old_targets is None:
            complete_psd_targets = targets
        else:
            complete_psd_targets = []
            for i, (p, t) in enumerate(zip(psd_targets, targets)):
                if memory_part[i]:
                    complete_psd_targets.append(
                        {'labels': t['labels'], 'masks': t['masks'], 'boxes': t['boxes']}
                    )
                else:
                    complete_psd_targets.append(
                        {
                            'labels': torch.cat([p['labels'], t['labels']]),\
                            'masks': torch.cat([p['masks'], t['masks']]),\
                            'boxes': torch.cat([p['boxes'], t['boxes']])
                        }
                    )


        # Retrieve the matching between the outputs of the last layer and the targets
        outputs_without_aux_no_fakeQuery = self._remove_fake_query(outputs_without_aux)
        indices = self.matcher(outputs_without_aux_no_fakeQuery, complete_psd_targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in complete_psd_targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'labels':
                new_indices, new_targets = self._modify_indices_targets_for_fake_query(indices, complete_psd_targets, fake_query_labels)
                losses.update(self.get_loss(loss, outputs, new_targets, new_indices, num_masks))
            elif loss == 'kd':
                # self.kd_type = 'l2'
                if not self.kd_deocder:
                    continue
                losses.update(self.get_loss(loss, outputs['distill_info'], old_outputs, self.kd_temperature2, num_masks))
            else:
                losses.update(self.get_loss(loss, outputs, complete_psd_targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs_no_fakeQuery = self._remove_fake_query(aux_outputs)
                indices = self.matcher(aux_outputs_no_fakeQuery, complete_psd_targets)
                for loss in self.losses:
                    if loss == 'labels':
                        new_indices, new_targets = self._modify_indices_targets_for_fake_query(indices, complete_psd_targets, fake_query_labels)
                        l_dict = self.get_loss(loss, aux_outputs, new_targets, new_indices, num_masks)
                    elif loss == 'kd':
                        continue
                        l_dict = self.get_loss(loss, outputs['distill_info']['aux_outputs'][i], old_outputs['aux_outputs'][i], indices, num_masks)
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, complete_psd_targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if "interm_outputs" in outputs:
            interm_outputs_no_fakeQuery = self._remove_fake_query(outputs["interm_outputs"])
            indices = self.matcher(interm_outputs_no_fakeQuery, complete_psd_targets)
            for loss in self.losses:
                if loss == 'labels':
                    new_indices, new_targets = self._modify_indices_targets_for_fake_query(indices, complete_psd_targets, fake_query_labels)
                    l_dict = self.get_loss(loss, outputs["interm_outputs"], new_targets, new_indices, num_masks)
                elif loss == 'kd':
                    # self.kd_type = 'kl'
                    # print(self.kd_temperature)
                    l_dict = self.get_loss(loss,outputs['distill_info']['interm_outputs'], old_outputs['interm_outputs'], self.kd_temperature, num_masks)
                else:
                    l_dict = self.get_loss(loss, outputs["interm_outputs"], complete_psd_targets, indices, num_masks)
                l_dict = {'interm_' + k: v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        return losses

    def L2_distillation_loss(self, inputs, targets):
        labels = targets.sigmoid()  # B x Q x C
        outputs = inputs.sigmoid()

        old_class_num = min(self.current_catagory_ids)
        labels = labels[..., :old_class_num]
        outputs = outputs[..., :old_class_num]
        batch_size = outputs.shape[0]
        loss = torch.pow((outputs - labels), 2).sum(dim=-1).mean()  # B

        return loss/batch_size

    def _remove_fake_query(self, outputs):
        ret = {}
        for k, v in outputs.items():
            fake_num = v.shape[1] - 100
            ret[k] = v[:, :-fake_num] if fake_num > 0 else v
        return ret
    
    def _modify_indices_targets_for_fake_query(self, indices, targets, fake_query_labels):
        if fake_query_labels is None:
            return indices, targets

        assert len(fake_query_labels) == len(targets) == len(indices)

        new_indices = []
        new_targets = copy.deepcopy(targets)
        for indice in indices:
            new_indice = list(copy.deepcopy(indice))
            new_indice[0] = torch.cat((new_indice[0], torch.arange(100,100+self.vq_number).to(torch.long))) # torch.tensor([100, 101, 102], device=indice[0].device)))
            max_indice = max(indice[1]).long() if len(indice[1]) > 0 else torch.tensor(-1).long()
            new_indice[1] = torch.cat(
                (
                    new_indice[1],
                    torch.tensor(
                        [max_indice + i + 1 for i in range(self.vq_number)], 
                        dtype=torch.long,  
                        device=indice[0].device 
                    )
                )
            )
            new_indices.append(new_indice)
        for i, (t, fql) in enumerate(zip(new_targets, fake_query_labels)):
            t['labels'] = torch.cat((t['labels'], fql.clone().detach().to(t['labels'].device)))
        
        return new_indices, new_targets
        
            
    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

