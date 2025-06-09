import copy
import itertools
import logging
import os
import weakref

from collections import OrderedDict
from tabulate import tabulate
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    HookBase,
    TrainerBase,
    create_ddp_model,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import setup_logger, _log_api_usage
from detectron2.engine import hooks
from fvcore.nn.precise_bn import get_bn_modules
# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

from .continual_panoptic_dataset_mapper import ContinualPanopticDatasetMapper
from .continual_semantic_dataset_mapper import ContinualSemanticDatasetMapper
from .continual_instance_dataset_mapper import ContinualInstanceDatasetMapper
from .evaluator import SemSegEvaluator, COCOPanopticEvaluator, InstanceSegEvaluator
from .train_loop import SimpleTrainer, AMPTrainer
import torch.distributed as dist
import collections
import functools

class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        if cfg.CONT.TASK > 1 and cfg.CONT.OLD_MODEL:
            cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
            cfg_old = cfg.clone()
            cfg_old.defrost()
            cfg_old.CONT.TASK = cfg.CONT.TASK - 1
            if cfg.CONT.TASK < 10:
                cfg_old.OUTPUT_DIR = cfg.OUTPUT_DIR[:-1] + f"{cfg_old.CONT.TASK}"
            else:
                cfg_old.OUTPUT_DIR = cfg.OUTPUT_DIR[:-2] + f"{cfg_old.CONT.TASK}"
            cfg_old.freeze()

            model_old = self.build_model(cfg_old).eval()
            model_old.load_state_dict(
                torch.load(cfg.CONT.OLD_WEIGHTS, map_location=torch.device("cpu"))["model"], strict=False)
            model_old = create_ddp_model(model_old, broadcast_buffers=False)
        else:
            for _ in range(5):
                print("No old model")
            model_old = None

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self.cfg = cfg
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, model_old, data_loader, optimizer, self.cfg.CONT.COLLECT_QUERY_MODE
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(
                SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder, cfg=cfg)
            )
        # panoptic segmentation
        if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder, cfg=cfg))
        # ADE20K
        # if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
        if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder, cfg=cfg))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "continual_panoptic":
            mapper = ContinualPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "continual_semantic":
            mapper = ContinualSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "continual_instance":
            mapper = ContinualInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger("detectron2.trainer")
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

                table_list = []
                if "sem_seg" in results_i.keys():
                    table_list.append([
                        "mIoU",
                        results_i["sem_seg"]["mIoU_old"],
                        results_i["sem_seg"]["mIoU_new"],
                        results_i["sem_seg"]["mIoU_past"],
                        results_i["sem_seg"]["mIoU_current"],
                        results_i["sem_seg"]["mIoU_all"]
                    ])
                    table_list.append([
                        "mACC",
                        results_i["sem_seg"]["mACC_old"],
                        results_i["sem_seg"]["mACC_new"],
                        results_i["sem_seg"]["mACC_past"],
                        results_i["sem_seg"]["mACC_current"],
                        results_i["sem_seg"]["mACC_all"]
                    ])

                if "segm" in results_i.keys():
                    table_list.append([
                        "AP",
                        results_i["segm"]["AP_old"],
                        results_i["segm"]["AP_new"],
                        results_i["segm"]["AP_past"],
                        results_i["segm"]["AP_current"],
                        results_i["segm"]["AP_all"]
                    ])

                if "panoptic_seg" in results_i.keys():
                    table_list.append([
                        "PQ",
                        results_i["panoptic_seg"]["PQ_Old"],
                        results_i["panoptic_seg"]["PQ_New"],
                        results_i["panoptic_seg"]["PQ_Past"],
                        results_i["panoptic_seg"]["PQ_Current"],
                        results_i["panoptic_seg"]["PQ_All"]
                    ])
                    table_list.append([
                        "PQ_th",
                        results_i["panoptic_seg"]["PQ_Old_th"],
                        results_i["panoptic_seg"]["PQ_New_th"],
                        results_i["panoptic_seg"]["PQ_Past_th"],
                        results_i["panoptic_seg"]["PQ_Current_th"],
                        results_i["panoptic_seg"]["PQ_All_th"]
                    ])
                    table_list.append([
                        "PQ_st",
                        results_i["panoptic_seg"]["PQ_Old_st"],
                        results_i["panoptic_seg"]["PQ_New_st"],
                        results_i["panoptic_seg"]["PQ_Past_st"],
                        results_i["panoptic_seg"]["PQ_Current_st"],
                        results_i["panoptic_seg"]["PQ_All_st"]
                    ])
                    table_list.append([
                        "SQ",
                        results_i["panoptic_seg"]["SQ_Old"],
                        results_i["panoptic_seg"]["SQ_New"],
                        results_i["panoptic_seg"]["SQ_Past"],
                        results_i["panoptic_seg"]["SQ_Current"],
                        results_i["panoptic_seg"]["SQ_All"]
                    ])
                    table_list.append([
                        "SQ_th",
                        results_i["panoptic_seg"]["SQ_Old_th"],
                        results_i["panoptic_seg"]["SQ_New_th"],
                        results_i["panoptic_seg"]["SQ_Past_th"],
                        results_i["panoptic_seg"]["SQ_Current_th"],
                        results_i["panoptic_seg"]["SQ_All_th"]
                    ])
                    table_list.append([
                        "SQ_st",
                        results_i["panoptic_seg"]["SQ_Old_st"],
                        results_i["panoptic_seg"]["SQ_New_st"],
                        results_i["panoptic_seg"]["SQ_Past_st"],
                        results_i["panoptic_seg"]["SQ_Current_st"],
                        results_i["panoptic_seg"]["SQ_All_st"]
                    ])
                    table_list.append([
                        "RQ",
                        results_i["panoptic_seg"]["RQ_Old"],
                        results_i["panoptic_seg"]["RQ_New"],
                        results_i["panoptic_seg"]["RQ_Past"],
                        results_i["panoptic_seg"]["RQ_Current"],
                        results_i["panoptic_seg"]["RQ_All"]
                    ])
                    table_list.append([
                        "RQ_th",
                        results_i["panoptic_seg"]["RQ_Old_th"],
                        results_i["panoptic_seg"]["RQ_New_th"],
                        results_i["panoptic_seg"]["RQ_Past_th"],
                        results_i["panoptic_seg"]["RQ_Current_th"],
                        results_i["panoptic_seg"]["RQ_All_th"]
                    ])
                    table_list.append([
                        "RQ_st",
                        results_i["panoptic_seg"]["RQ_Old_st"],
                        results_i["panoptic_seg"]["RQ_New_st"],
                        results_i["panoptic_seg"]["RQ_Past_st"],
                        results_i["panoptic_seg"]["RQ_Current_st"],
                        results_i["panoptic_seg"]["RQ_All_st"]
                    ])

                table_headers = ["", "old", "new", "past", "current", "all"]
                table = tabulate(table_list, headers=table_headers,
                                 tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center")
                logger.info("Result Summary:\n" + table)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    # We need to collect data from all processes for fake query
    def after_train(self):
        logger = logging.getLogger("detectron2.trainer")
        self.storage.iter = self.iter
        if self.cfg.CONT.COLLECT_QUERY_MODE and self.iter == self.cfg.SOLVER.MAX_ITER:
            import json
            root = self.cfg.OUTPUT_DIR
            file = os.path.join(root, "psd_distribution.json")
            if not os.path.exists(file):
                os.makedirs(os.path.dirname(file), exist_ok=True)
            save = self.model.psd_num

            if self.cfg.CONT.CUMULATIVE_PSDNUM == True and self.cfg.CONT.TASK > 2:
                old_root = root[:-1] + str(self.cfg.CONT.TASK-1)
                old_file = os.path.join(old_root, "psd_distribution.json")
                old_save = json.load(open(old_file, 'r'))
                save += torch.tensor(old_save)
            with open(file, 'w+') as f:
                json.dump(save.tolist(), f)
                logger.info("Save psd_distribution.json to {}".format(file))
                exit()
        elif self.iter == self.cfg.SOLVER.MAX_ITER:
            collect = self.model.module.collect  # collec:dict, key: deque
            if dist.is_initialized():
                print("Dist initialized, gathering collect data...")
                # Collect data from all processes
                gathered_collect = [None for _ in range(dist.get_world_size())]
                print("Gathered collect data, combining...")
                dist.all_gather_object(gathered_collect, collect)

            if dist.get_rank() == 0:
                # combine collect
                deque_factory_with_size = functools.partial(deque_factory, self.cfg.CONT.LIB_SIZE)
                combined_collect = collections.defaultdict(deque_factory_with_size)
                
                for gpu_collect in gathered_collect:
                    for key, deque_value in gpu_collect.items():
                        combined_collect[key].extend(deque_value)

                file = os.path.join(self.cfg.OUTPUT_DIR, "fake_query.pkl")
                torch.save(combined_collect, file)
                # with open(file, 'wb') as f:
                #     pickle.dump(combined_collect, f)
                logger.info(f"Save fake_query.pkl to {file}")

            # file = os.path.join(self.cfg.OUTPUT_DIR, "fake_query.pkl")
            # with open(file, 'wb') as f:
                # pickle.dump(self.model.module.collect, f)
        if not self.cfg.CONT.VQ_STORE:
            for h in self._hooks:
                h.after_train()

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            (
                hooks.PreciseBN(
                    # Run at the same freq as (but before) evaluation.
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    # Build a new data loader to not affect training
                    self.build_train_loader(cfg),
                    cfg.TEST.PRECISE_BN.NUM_ITER,
                )
                if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
                else None
            ),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process() and not cfg.CONT.COLLECT_QUERY_MODE:
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

def deque_factory(maxlen=80):
    return collections.deque(maxlen=maxlen)