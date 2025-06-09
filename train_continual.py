# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import sys
sys.path.insert(0, '/public/home/zhuyuchen530/projects/cvpr24/fake3')
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import os
import math

from detectron2.data import DatasetCatalog, MetadataCatalog

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import verify_results
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import add_maskformer2_config
from mask2former.data.datasets.register_ade20k_panoptic import (
    register_current_ade20k_panoptic,
    register_complete_ade20k_sem,
    register_mem_ade20k_sem
)
from mask2former.data.datasets.register_ade20k_instance import register_current_ade20k_instance

from continual import add_continual_config, Trainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_continual_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if cfg.CONT.TASK > 1:
        if cfg.CONT.WEIGHTS is not None:
            cfg.MODEL.WEIGHTS = cfg.CONT.WEIGHTS
        else:
            if args.eval_only:
                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR[:-1] + str(cfg.CONT.TASK), "model_final.pth")
                if cfg.CONT.TASK >= 10:
                    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR[:-2] + str(cfg.CONT.TASK), "model_final.pth")
            else:
                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR[:-1] + str(cfg.CONT.TASK - 1), "model_final.pth")
                if cfg.CONT.TASK >= 10:
                    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR[:-2] + str(cfg.CONT.TASK - 1), "model_final.pth")

        if cfg.CONT.OLD_WEIGHTS is None:
            cfg.CONT.OLD_WEIGHTS = cfg.MODEL.WEIGHTS

    elif args.eval_only:
        if cfg.CONT.WEIGHTS is not None:
            cfg.MODEL.WEIGHTS = cfg.CONT.WEIGHTS
        else:
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR[:-1] + str(cfg.CONT.TASK), "model_final.pth")
            if cfg.CONT.TASK >= 10:
                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR[:-2] + str(cfg.CONT.TASK), "model_final.pth")

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    # panoptic segmentation
    if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON == True:
        predefined_split = {
            "current_ade20k_panoptic_train": (
                "datasets/ADEChallengeData2016/images/training",
                "datasets/ADEChallengeData2016/ade20k_panoptic_train",
                f"json/pan/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_pan.json",
                "datasets/ADEChallengeData2016/annotations_detectron2/training",
                f"json/pan/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_inst.json",
            ),
            "current_ade20k_panoptic_val": (
                "datasets/ADEChallengeData2016/images/validation",
                "datasets/ADEChallengeData2016/ade20k_panoptic_val",
                f"json/pan/val_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_pan.json",
                "datasets/ADEChallengeData2016/annotations_detectron2/validation",
                f"json/pan/val_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_inst.json",
            ),
        }
        register_current_ade20k_panoptic(predefined_split)

        if cfg.CONT.TASK > 1 and cfg.CONT.MEMORY == True:
            print('***********Including Memory***********')
            predefined_split_memory = {
                "memory_ade20k_panoptic_train": (
                    "datasets/ADEChallengeData2016/images/training",
                    "datasets/ADEChallengeData2016/ade20k_panoptic_train",
                    f"json/memory/pan/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK - 1}_pan.json",
                    "datasets/ADEChallengeData2016/annotations_detectron2/training",
                    f"json/memory/pan/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK - 1}_inst.json",
                ),
            }
            register_current_ade20k_panoptic(predefined_split_memory)

            current_pan_data_train = DatasetCatalog.get("current_ade20k_panoptic_train")
            memory_pan_data_train = DatasetCatalog.get("memory_ade20k_panoptic_train")

            cur_w_mem_pan_data_train = current_pan_data_train + memory_pan_data_train
            cur_w_mem_pan_meta_train = MetadataCatalog.get("current_ade20k_panoptic_train")

            DatasetCatalog.register(
                "cur_w_mem_ade20k_panoptic_train", lambda: cur_w_mem_pan_data_train
            )
            MetadataCatalog.get("cur_w_mem_ade20k_panoptic_train").set(
                evaluator_type=cur_w_mem_pan_meta_train.evaluator_type,
                ignore_label=cur_w_mem_pan_meta_train.ignore_label,
                image_root=cur_w_mem_pan_meta_train.image_root,
                json_file=cur_w_mem_pan_meta_train.json_file,
                label_divisor=cur_w_mem_pan_meta_train.label_divisor,
                stuff_classes=cur_w_mem_pan_meta_train.stuff_classes,
                stuff_colors=cur_w_mem_pan_meta_train.stuff_colors,
                stuff_dataset_id_to_contiguous_id=cur_w_mem_pan_meta_train.stuff_dataset_id_to_contiguous_id,
                thing_classes=cur_w_mem_pan_meta_train.thing_classes,
                thing_colors=cur_w_mem_pan_meta_train.thing_colors,
                thing_dataset_id_to_contiguous_id=cur_w_mem_pan_meta_train.thing_dataset_id_to_contiguous_id,
            )
            cfg.defrost()
            cfg.DATASETS.TRAIN = ("cur_w_mem_ade20k_panoptic_train",)
            cfg.freeze()

    # semantic segmentation
    if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON == False and cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON == False:
        register_complete_ade20k_sem("datasets")
        complete_sem_data_train = DatasetCatalog.get("complete_ade20k_sem_seg_train")
        complete_sem_data_val = DatasetCatalog.get("complete_ade20k_sem_seg_val")
        img_list_train = [img['file_name'] for img in DatasetCatalog.get("current_ade20k_panoptic_train")]
        img_list_val = [img['file_name'] for img in DatasetCatalog.get("current_ade20k_panoptic_val")]
        current_sem_data_train = []
        current_sem_data_val = []
        for img in complete_sem_data_train:
            if img['file_name'] in img_list_train:
                current_sem_data_train.append(img)
        for img in complete_sem_data_val:
            if img['file_name'] in img_list_val:
                current_sem_data_val.append(img)
        complete_sem_meta_train = MetadataCatalog.get("complete_ade20k_sem_seg_train")
        complete_sem_meta_val = MetadataCatalog.get("complete_ade20k_sem_seg_val")

        DatasetCatalog.register(
            "current_ade20k_sem_seg_train", lambda: current_sem_data_train
        )
        DatasetCatalog.register(
            "current_ade20k_sem_seg_val", lambda: current_sem_data_val
        )
        MetadataCatalog.get("current_ade20k_sem_seg_train").set(
            evaluator_type=complete_sem_meta_train.evaluator_type,
            ignore_label=complete_sem_meta_train.ignore_label,
            image_root=complete_sem_meta_train.image_root,
            sem_seg_root=complete_sem_meta_train.sem_seg_root,
            stuff_classes=complete_sem_meta_train.stuff_classes,
        )
        MetadataCatalog.get("current_ade20k_sem_seg_val").set(
            evaluator_type=complete_sem_meta_val.evaluator_type,
            ignore_label=complete_sem_meta_val.ignore_label,
            image_root=complete_sem_meta_val.image_root,
            sem_seg_root=complete_sem_meta_val.sem_seg_root,
            stuff_classes=complete_sem_meta_val.stuff_classes,
        )

        if cfg.CONT.TASK > 1 and cfg.CONT.MEMORY == True:
            register_mem_ade20k_sem(
                "datasets",
                f"json/memory/sem/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK - 1}"
            )
            current_sem_data_train = DatasetCatalog.get("current_ade20k_sem_seg_train")
            current_sem_meta_train = MetadataCatalog.get("current_ade20k_sem_seg_train")
            memory_sem_data_train = DatasetCatalog.get("memory_ade20k_sem_seg_train")

            for per_mem_data in current_sem_data_train:
                per_mem_data['memory'] = False
            for per_mem_data in memory_sem_data_train:
                per_mem_data['memory'] = True

            cur_w_mem_sem_data_train = current_sem_data_train + memory_sem_data_train
            DatasetCatalog.register(
                "cur_w_mem_sem_data_train", lambda: cur_w_mem_sem_data_train
            )
            MetadataCatalog.get("cur_w_mem_sem_data_train").set(
                evaluator_type=current_sem_meta_train.evaluator_type,
                ignore_label=current_sem_meta_train.ignore_label,
                image_root=current_sem_meta_train.image_root,
                stuff_classes=current_sem_meta_train.stuff_classes,
            )
            cfg.defrost()
            cfg.DATASETS.TRAIN = ("cur_w_mem_sem_data_train",)
            cfg.freeze()

    # instance segmentation
    if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON == False and cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON == False:

        predefined_split = {
            "current_ade20k_instance_train": (
                "datasets/ADEChallengeData2016/images/training",
                f"json/inst/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_inst.json",
            ),
            "current_ade20k_instance_val": (
                "datasets/ADEChallengeData2016/images/validation",
                f"json/inst/val_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK}_inst.json",
            ),
        }
        register_current_ade20k_instance(predefined_split)

        if cfg.CONT.TASK > 1 and cfg.CONT.MEMORY == True:
            predefined_split_memory = {
                "memory_ade20k_instance_train": (
                    "datasets/ADEChallengeData2016/images/training",
                    f"json/memory/inst/train_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}_step{cfg.CONT.TASK - 1}_inst.json",
                ),
            }
            register_current_ade20k_instance(predefined_split_memory)

            current_inst_data_train = DatasetCatalog.get("current_ade20k_instance_train")
            current_inst_meta_train = MetadataCatalog.get("current_ade20k_instance_train")
            memory_inst_data_train = DatasetCatalog.get("memory_ade20k_instance_train")

            cur_w_mem_inst_data_train = current_inst_data_train + memory_inst_data_train
            DatasetCatalog.register(
                "cur_w_mem_ade20k_instance_train", lambda: cur_w_mem_inst_data_train
            )
            MetadataCatalog.get("cur_w_mem_ade20k_instance_train").set(
                evaluator_type=current_inst_meta_train.evaluator_type,
                image_root=current_inst_meta_train.image_root,
                # json_file=cur_w_mem_inst_meta_train.json_file,
                name="cur_w_mem_ade20k_instance_train",
                thing_classes=current_inst_meta_train.thing_classes,
                thing_dataset_id_to_contiguous_id=current_inst_meta_train.thing_dataset_id_to_contiguous_id,
            )
            cfg.defrost()
            cfg.DATASETS.TRAIN = ("cur_w_mem_ade20k_instance_train",)
            cfg.freeze()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
