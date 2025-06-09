# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import DatasetEvaluator, SemSegEvaluator, COCOPanopticEvaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco, create_small_table
from detectron2.evaluation.panoptic_evaluation import _print_panoptic_results
import logging
import pycocotools.mask as mask_util

logger = logging.getLogger('detectron2')

class SemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        cfg=None,
    ):
        super().__init__(dataset_name, distributed, output_dir, num_classes=num_classes, ignore_label=ignore_label)

        tot_cls = cfg.CONT.TOT_CLS
        base_cls = cfg.CONT.BASE_CLS
        inc_cls = cfg.CONT.INC_CLS
        task = cfg.CONT.TASK

        num_tasks = 1 + (tot_cls - base_cls) // inc_cls
        n_cls_in_tasks = [base_cls] + [inc_cls] * (num_tasks - 1)
        self.category_ids_old = list(range(base_cls))
        self.category_ids_new = list(range(base_cls, sum(n_cls_in_tasks[:task])))
        self.category_ids_past = list(range(sum(n_cls_in_tasks[:task - 1])))
        self.category_ids_current = list(range(sum(n_cls_in_tasks[:task - 1]), sum(n_cls_in_tasks[:task])))
        self.category_ids_all = list(range(sum(n_cls_in_tasks[:task])))

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int32)
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=np.int32)

            for cls in set(np.unique(gt)) - set(self.category_ids_all) - set([self._ignore_label]):
            # for cls in set(np.unique(gt)) - set(range(100)) - set([self._ignore_label]):
                gt[gt == cls] = self._ignore_label

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float32)
        iou = np.full(self._num_classes, np.nan, dtype=np.float32)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float32)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float32)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float32)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        category_ids_old_and_acc_valid = \
            list(np.array(self.category_ids_old)[acc_valid[self.category_ids_old]])
        category_ids_new_and_acc_valid = \
            list(np.array(self.category_ids_new)[acc_valid[self.category_ids_new]])
        category_ids_past_and_acc_valid = \
            list(np.array(self.category_ids_past)[acc_valid[self.category_ids_past]])
        category_ids_current_and_acc_valid = \
            list(np.array(self.category_ids_current)[acc_valid[self.category_ids_current]])
        category_ids_all_and_acc_valid = \
            list(np.array(self.category_ids_all)[acc_valid[self.category_ids_all]])

        miou_old = np.sum(iou[category_ids_old_and_acc_valid]) / len(category_ids_old_and_acc_valid)
        miou_new = np.sum(iou[category_ids_new_and_acc_valid]) / len(category_ids_new_and_acc_valid)
        miou_past = np.sum(iou[category_ids_past_and_acc_valid]) / len(category_ids_past_and_acc_valid)
        miou_current = np.sum(iou[category_ids_current_and_acc_valid]) / len(category_ids_current_and_acc_valid)
        miou_all = np.sum(iou[category_ids_all_and_acc_valid]) / len(category_ids_all_and_acc_valid)

        macc_old = np.sum(acc[category_ids_old_and_acc_valid]) / len(category_ids_old_and_acc_valid)
        macc_new = np.sum(acc[category_ids_new_and_acc_valid]) / len(category_ids_new_and_acc_valid)
        macc_past = np.sum(acc[category_ids_past_and_acc_valid]) / len(category_ids_past_and_acc_valid)
        macc_current = np.sum(acc[category_ids_current_and_acc_valid]) / len(category_ids_current_and_acc_valid)
        macc_all = np.sum(acc[category_ids_all_and_acc_valid]) / len(category_ids_all_and_acc_valid)

        # res = {}
        # res["mIoU"] = 100 * miou
        # res["fwIoU"] = 100 * fiou
        # for i, name in enumerate(self._class_names):
        #     res["IoU-{}".format(name)] = 100 * iou[i]
        # res["mACC"] = 100 * macc
        # res["pACC"] = 100 * pacc
        # for i, name in enumerate(self._class_names):
        #     res["ACC-{}".format(name)] = 100 * acc[i]

        logger.info(f'{category_ids_all_and_acc_valid}')
        logger.info(f'{len(category_ids_all_and_acc_valid)}')
        for i, x in enumerate(iou):
            logger.info(f'category {i} iou: {x}')
        res = {}
        res["mIoU"] = 100 * miou
        res["mIoU_old"] = 100 * miou_old
        res["mIoU_new"] = 100 * miou_new
        res["mIoU_past"] = 100 * miou_past
        res["mIoU_current"] = 100 * miou_current
        res["mIoU_all"] = 100 * miou_all

        res["mACC"] = 100 * macc
        res["mACC_old"] = 100 * macc_old
        res["mACC_new"] = 100 * macc_new
        res["mACC_past"] = 100 * macc_past
        res["mACC_current"] = 100 * macc_current
        res["mACC_all"] = 100 * macc_all

        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
    
    def encode_json_sem_seg(self, sem_seg, input_file_name):
            """
            Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
            See http://cocodataset.org/#format-results
            """
            json_list = []
            for label in np.unique(sem_seg):
                if self._contiguous_id_to_dataset_id is not None:
                    # assert (
                    #     label in self._contiguous_id_to_dataset_id
                    # ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                    if label in self._contiguous_id_to_dataset_id:
                        dataset_id = self._contiguous_id_to_dataset_id[label]
                    else:
                        dataset_id = -1
                else:
                    dataset_id = int(label)
                mask = (sem_seg == label).astype(np.uint8)
                mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
                mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
                json_list.append(
                    {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
                )
            return json_list

class COCOPanopticEvaluator(COCOPanopticEvaluator):

    def __init__(self, dataset_name, output_dir=None, cfg=None):
        super().__init__(dataset_name, output_dir)

        tot_cls = cfg.CONT.TOT_CLS
        base_cls = cfg.CONT.BASE_CLS
        inc_cls = cfg.CONT.INC_CLS
        task = cfg.CONT.TASK

        num_tasks = 1 + (tot_cls - base_cls) // inc_cls
        n_cls_in_tasks = [base_cls] + [inc_cls] * (num_tasks - 1)
        self.category_ids_old = list(range(base_cls))
        self.category_ids_new = list(range(base_cls, sum(n_cls_in_tasks[:task])))
        self.category_ids_past = list(range(sum(n_cls_in_tasks[:task - 1])))
        self.category_ids_current = list(range(sum(n_cls_in_tasks[:task - 1]), sum(n_cls_in_tasks[:task])))
        self.category_ids_all = list(range(sum(n_cls_in_tasks[:task])))
        self.n_cls_in_tasks_cumsum = [0] + torch.as_tensor(n_cls_in_tasks).cumsum(0).tolist()

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            # logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        categories = {el['id']: el for el in json_data['categories']}
        valid_ids = [
            k for k, v in pq_res["per_class"].items()
            if v != {'pq': 0.0, 'sq': 0.0, 'rq': 0.0} or type(v['sq']) != float
        ]
        thing_ids = [k for k, v in categories.items() if v['isthing'] == 1]
        stuff_ids = [k for k, v in categories.items() if v['isthing'] == 0]

        # compute ids
        old_ids = set(self.category_ids_old) & set(valid_ids)
        new_ids = set(self.category_ids_new) & set(valid_ids)
        past_ids = set(self.category_ids_past) & set(valid_ids)
        current_ids = set(self.category_ids_current) & set(valid_ids)
        all_ids = set(self.category_ids_all) & set(valid_ids)

        old_th_ids = set(self.category_ids_old) & set(thing_ids) & set(valid_ids)
        new_th_ids = set(self.category_ids_new) & set(thing_ids) & set(valid_ids)
        past_th_ids = set(self.category_ids_past) & set(thing_ids) & set(valid_ids)
        current_th_ids = set(self.category_ids_current) & set(thing_ids) & set(valid_ids)
        all_th_ids = set(self.category_ids_all) & set(thing_ids) & set(valid_ids)

        old_st_ids = set(self.category_ids_old) & set(stuff_ids) & set(valid_ids)
        new_st_ids = set(self.category_ids_new) & set(stuff_ids) & set(valid_ids)
        past_st_ids = set(self.category_ids_past) & set(stuff_ids) & set(valid_ids)
        current_st_ids = set(self.category_ids_current) & set(stuff_ids) & set(valid_ids)
        all_st_ids = set(self.category_ids_all) & set(stuff_ids) & set(valid_ids)

        from rich import print
        select_ids = [35, 37, 44, 46, 49, 50, 51, 56, 58, 60, 64, 65, 68, 70, 71, 73, 77,78, 79, 81, 91, 92, 96, 97, 99]
        # select_ids= [ 6, 13, 20, 21, 29, 31, 34, 38, 40, 45, 46, 48, 51, 52, 54, 55, 58, 59,
        #  60, 61, 65, 68, 70, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 90,
        #  91, 92, 93, 94, 96, 99]
        not_selected = [i for i in range(100) if i not in select_ids]
        for i, p in enumerate(pq_res['per_class'].values()):
            logger.info(f'category {i} PQ: {p["pq"] * 100:.3f} SQ: {p["sq"] * 100:.3f} RQ: {p["rq"] * 100:.3f}')
        select_pq = np.sum([pq_res['per_class'][i]['pq'] * 100 for i in select_ids])/25
        logger.info(f'PQ of select categories: {select_pq}')
        not_select_pq = np.sum([pq_res['per_class'][i]['pq'] * 100 for i in not_selected])/75
        logger.info(f'PQ of not select base categories: {not_select_pq}')
        # compute pq, sq, rq
        pq_everystep = np.array([])
        for i in range(len(self.n_cls_in_tasks_cumsum)-1):
            step_mean = np.mean([pq_res["per_class"][k]["pq"] for k in range(self.n_cls_in_tasks_cumsum[i], self.n_cls_in_tasks_cumsum[i+1])])
            pq_everystep = np.append(pq_everystep, step_mean)
        sq_everystep = np.array([])
        for i in range(len(self.n_cls_in_tasks_cumsum)-1):
            step_mean = np.mean([pq_res["per_class"][k]["sq"] for k in range(self.n_cls_in_tasks_cumsum[i], self.n_cls_in_tasks_cumsum[i+1])])
            sq_everystep = np.append(sq_everystep, step_mean)
        rq_everystep = np.array([])
        for i in range(len(self.n_cls_in_tasks_cumsum)-1):
            step_mean = np.mean([pq_res["per_class"][k]["rq"] for k in range(self.n_cls_in_tasks_cumsum[i], self.n_cls_in_tasks_cumsum[i+1])])
            rq_everystep = np.append(rq_everystep, step_mean)
        
        stage2PQ = [pq_res["per_class"][k]["pq"] for k in range(100, 105)]

        pq_old = np.mean([pq_res["per_class"][k]["pq"] for k in old_ids])
        pq_new = np.mean([pq_res["per_class"][k]["pq"] for k in new_ids])
        pq_past = np.mean([pq_res["per_class"][k]["pq"] for k in past_ids])
        pq_current = np.mean([pq_res["per_class"][k]["pq"] for k in current_ids])
        pq_all = np.mean([pq_res["per_class"][k]["pq"] for k in all_ids])

        sq_old = np.mean([pq_res["per_class"][k]["sq"] for k in old_ids])
        sq_new = np.mean([pq_res["per_class"][k]["sq"] for k in new_ids])
        sq_past = np.mean([pq_res["per_class"][k]["sq"] for k in past_ids])
        sq_current = np.mean([pq_res["per_class"][k]["sq"] for k in current_ids])
        sq_all = np.mean([pq_res["per_class"][k]["sq"] for k in all_ids])

        rq_old = np.mean([pq_res["per_class"][k]["rq"] for k in old_ids])
        rq_new = np.mean([pq_res["per_class"][k]["rq"] for k in new_ids])
        rq_past = np.mean([pq_res["per_class"][k]["rq"] for k in past_ids])
        rq_current = np.mean([pq_res["per_class"][k]["rq"] for k in current_ids])
        rq_all = np.mean([pq_res["per_class"][k]["rq"] for k in all_ids])

        # compute pq, sq, rq for thing
        pq_old_th = np.mean([pq_res["per_class"][k]["pq"] for k in old_th_ids])
        pq_new_th = np.mean([pq_res["per_class"][k]["pq"] for k in new_th_ids])
        pq_past_th = np.mean([pq_res["per_class"][k]["pq"] for k in past_th_ids])
        pq_current_th = np.mean([pq_res["per_class"][k]["pq"] for k in current_th_ids])
        pq_all_th = np.mean([pq_res["per_class"][k]["pq"] for k in all_th_ids])

        sq_old_th = np.mean([pq_res["per_class"][k]["sq"] for k in old_th_ids])
        sq_new_th = np.mean([pq_res["per_class"][k]["sq"] for k in new_th_ids])
        sq_past_th = np.mean([pq_res["per_class"][k]["sq"] for k in past_th_ids])
        sq_current_th = np.mean([pq_res["per_class"][k]["sq"] for k in current_th_ids])
        sq_all_th = np.mean([pq_res["per_class"][k]["sq"] for k in all_th_ids])

        rq_old_th = np.mean([pq_res["per_class"][k]["rq"] for k in old_th_ids])
        rq_new_th = np.mean([pq_res["per_class"][k]["rq"] for k in new_th_ids])
        rq_past_th = np.mean([pq_res["per_class"][k]["rq"] for k in past_th_ids])
        rq_current_th = np.mean([pq_res["per_class"][k]["rq"] for k in current_th_ids])
        rq_all_th = np.mean([pq_res["per_class"][k]["rq"] for k in all_th_ids])

        # compute pq, sq, rq for stuff
        pq_old_st = np.mean([pq_res["per_class"][k]["pq"] for k in old_st_ids])
        pq_new_st = np.mean([pq_res["per_class"][k]["pq"] for k in new_st_ids])
        pq_past_st = np.mean([pq_res["per_class"][k]["pq"] for k in past_st_ids])
        pq_current_st = np.mean([pq_res["per_class"][k]["pq"] for k in current_st_ids])
        pq_all_st = np.mean([pq_res["per_class"][k]["pq"] for k in all_st_ids])

        sq_old_st = np.mean([pq_res["per_class"][k]["sq"] for k in old_st_ids])
        sq_new_st = np.mean([pq_res["per_class"][k]["sq"] for k in new_st_ids])
        sq_past_st = np.mean([pq_res["per_class"][k]["sq"] for k in past_st_ids])
        sq_current_st = np.mean([pq_res["per_class"][k]["sq"] for k in current_st_ids])
        sq_all_st = np.mean([pq_res["per_class"][k]["sq"] for k in all_st_ids])

        rq_old_st = np.mean([pq_res["per_class"][k]["rq"] for k in old_st_ids])
        rq_new_st = np.mean([pq_res["per_class"][k]["rq"] for k in new_st_ids])
        rq_past_st = np.mean([pq_res["per_class"][k]["rq"] for k in past_st_ids])
        rq_current_st = np.mean([pq_res["per_class"][k]["rq"] for k in current_st_ids])
        rq_all_st = np.mean([pq_res["per_class"][k]["rq"] for k in all_st_ids])

        # res = {}
        # res["PQ"] = 100 * pq_res["All"]["pq"]
        # res["SQ"] = 100 * pq_res["All"]["sq"]
        # res["RQ"] = 100 * pq_res["All"]["rq"]
        # res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        # res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        # res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        # res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        # res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        # res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        res = {}

        res["PQ_Old"] = 100 * pq_old
        res["SQ_Old"] = 100 * sq_old
        res["RQ_Old"] = 100 * rq_old
        res["PQ_New"] = 100 * pq_new
        res["SQ_New"] = 100 * sq_new
        res["RQ_New"] = 100 * rq_new
        res["PQ_Past"] = 100 * pq_past
        res["SQ_Past"] = 100 * sq_past
        res["RQ_Past"] = 100 * rq_past
        res["PQ_Current"] = 100 * pq_current
        res["SQ_Current"] = 100 * sq_current
        res["RQ_Current"] = 100 * rq_current
        res["PQ_All"] = 100 * pq_all
        res["SQ_All"] = 100 * sq_all
        res["RQ_All"] = 100 * rq_all

        res["PQ_Old_th"] = 100 * pq_old_th
        res["SQ_Old_th"] = 100 * sq_old_th
        res["RQ_Old_th"] = 100 * rq_old_th
        res["PQ_New_th"] = 100 * pq_new_th
        res["SQ_New_th"] = 100 * sq_new_th
        res["RQ_New_th"] = 100 * rq_new_th
        res["PQ_Past_th"] = 100 * pq_past_th
        res["SQ_Past_th"] = 100 * sq_past_th
        res["RQ_Past_th"] = 100 * rq_past_th
        res["PQ_Current_th"] = 100 * pq_current_th
        res["SQ_Current_th"] = 100 * sq_current_th
        res["RQ_Current_th"] = 100 * rq_current_th
        res["PQ_All_th"] = 100 * pq_all_th
        res["SQ_All_th"] = 100 * sq_all_th
        res["RQ_All_th"] = 100 * rq_all_th

        res["PQ_Old_st"] = 100 * pq_old_st
        res["SQ_Old_st"] = 100 * sq_old_st
        res["RQ_Old_st"] = 100 * rq_old_st
        res["PQ_New_st"] = 100 * pq_new_st
        res["SQ_New_st"] = 100 * sq_new_st
        res["RQ_New_st"] = 100 * rq_new_st
        res["PQ_Past_st"] = 100 * pq_past_st
        res["SQ_Past_st"] = 100 * sq_past_st
        res["RQ_Past_st"] = 100 * rq_past_st
        res["PQ_Current_st"] = 100 * pq_current_st
        res["SQ_Current_st"] = 100 * sq_current_st
        res["RQ_Current_st"] = 100 * rq_current_st
        res["PQ_All_st"] = 100 * pq_all_st
        res["SQ_All_st"] = 100 * sq_all_st
        res["RQ_All_st"] = 100 * rq_all_st

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)
        logger.info([f"step {i+1}: {pq * 100:.3f} " for i, pq in enumerate(pq_everystep)])
        logger.info([f"step {i+1}: {sq * 100:.3f} " for i, sq in enumerate(sq_everystep)])
        logger.info([f"step {i+1}: {rq * 100:.3f} " for i, rq in enumerate(rq_everystep)])
        logger.info([f"class{99+i}_PQ: {pq * 100:.3f} " for i, pq in enumerate(stage2PQ)])

        return results


class InstanceSegEvaluator(COCOEvaluator):

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        cfg=None,
    ):
        super().__init__(
            dataset_name,
            tasks,
            distributed,
            output_dir,
            max_dets_per_image=max_dets_per_image,
            use_fast_impl=use_fast_impl,
            kpt_oks_sigmas=kpt_oks_sigmas
        )

        tot_cls = cfg.CONT.TOT_CLS
        base_cls = cfg.CONT.BASE_CLS
        inc_cls = cfg.CONT.INC_CLS
        task = cfg.CONT.TASK

        num_tasks = 1 + (tot_cls - base_cls) // inc_cls
        n_cls_in_tasks = [base_cls] + [inc_cls] * (num_tasks - 1)

        if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON == False and cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON == False:
            inst_ids = [7, 8, 10, 12, 14, 15, 18, 19, 20, 22, 23, 24, 27, 30, 31, 32, 33, 35, 36, 37, 38, 39, 41, 42,
                        43, 44, 45, 47, 49, 50, 53, 55, 56, 57, 58, 62, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76,
                        78, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 92, 93, 95, 97, 98, 102, 103, 104, 107, 108, 110,
                        111, 112, 115, 116, 118, 119, 120, 121, 123, 124, 125, 126, 127, 129, 130, 132, 133, 134, 135,
                        136, 137, 138, 139, 142, 143, 144, 146, 147, 148, 149]
            inst_ids = np.array(inst_ids)
            self.category_ids_old = list(inst_ids[list(range(base_cls))])
            self.category_ids_new = list(inst_ids[list(range(base_cls, sum(n_cls_in_tasks[:task])))])
            self.category_ids_past = list(inst_ids[list(range(sum(n_cls_in_tasks[:task - 1])))])
            self.category_ids_current = list(inst_ids[list(range(sum(n_cls_in_tasks[:task - 1]), sum(n_cls_in_tasks[:task])))])
            self.category_ids_all = list(inst_ids[list(range(sum(n_cls_in_tasks[:task])))])
            self.category_ids_whole_dataset = list(inst_ids[list(range(tot_cls))])
        else:
            self.category_ids_old = list(range(base_cls))
            self.category_ids_new = list(range(base_cls, sum(n_cls_in_tasks[:task])))
            self.category_ids_past = list(range(sum(n_cls_in_tasks[:task - 1])))
            self.category_ids_current = list(range(sum(n_cls_in_tasks[:task - 1]), sum(n_cls_in_tasks[:task])))
            self.category_ids_all = list(range(sum(n_cls_in_tasks[:task])))
            self.category_ids_whole_dataset = list(range(tot_cls))

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        # tasks = self._tasks or self._tasks_from_predictions(coco_results)
        tasks = ['segm']

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            # all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            # num_classes = len(all_contiguous_ids)
            # assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                # assert category_id < num_classes, (
                #     f"A prediction has class={category_id}, "
                #     f"but the dataset only has {num_classes} classes and "
                #     f"predicted class id should be in [0, {num_classes - 1}]."
                # )
                assert category_id in reverse_id_mapping, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has class ids in {dataset_id_to_contiguous_id}."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    # use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            res_per_class = [v for k, v in res.items() if k not in ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']]
            ap_ids_mapped = {k: np.nan for k in self.category_ids_whole_dataset}

            for i, k in enumerate(dataset_id_to_contiguous_id.keys()):
                ap_ids_mapped[k] = res_per_class[i]

            ap_old = np.nanmean([ap_ids_mapped[k] for k in self.category_ids_old])
            ap_new = np.nanmean([ap_ids_mapped[k] for k in self.category_ids_new])
            ap_past = np.nanmean([ap_ids_mapped[k] for k in self.category_ids_past])
            ap_current = np.nanmean([ap_ids_mapped[k] for k in self.category_ids_current])
            ap_all = np.nanmean([ap_ids_mapped[k] for k in self.category_ids_all])

            res['AP_old'] = ap_old
            res['AP_new'] = ap_new
            res['AP_past'] = ap_past
            res['AP_current'] = ap_current
            res['AP_all'] = ap_all

            self._results[task] = res