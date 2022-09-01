import os
import logging
import weakref
import torch
from collections import OrderedDict

from detectron2.engine.defaults import DefaultTrainer, TrainerBase, hooks
from detectron2.utils.logger import setup_logger
from detectron2.engine import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.evaluation import verify_results
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter

from adet.checkpoint import AdetCheckpointer
from adet.data.dataset_mapper import DatasetMapperWithBasis
from detectron2.modeling import GeneralizedRCNNWithTTA
from adet.evaluation import AmodalVisibleEvaluator, VisibleEvaluator, AmodalEvaluator
from adet.evaluation.evaluator import DatasetEvaluators, DatasetEvaluator, inference_on_dataset
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    print_csv_format,
    verify_results,
)

from fvcore.common.config import CfgNode
from adet.modeling.domain_shift_modules import (
    # FusionDiscriminator,
    StudentAccusingDiscriminator,
)
import mycode.tools.det2_adet_subs.hooks as myhooks
import mycode.tools.det2_adet_subs.events as myevents
from mycode.tools import cfg_force_merge
from mycode.tools import is_in_colab, styler

from .subtrainer import DASubTrainer


__all__ = [
    "DATrainer"
]


class DATrainer(DefaultTrainer):
    def __init__(self, cfg: CfgNode, resume=False):
        TrainerBase.__init__(self)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        logger.info(styler.bold_pink + "Using \"StudentAccusingDA\" trainer..." + styler.reset)
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Create other cfg
        cfg_sup = cfg_force_merge(cfg, cfg.DOMAIN_ADAPTATION.TARGET_DATASET_CONFIG)
        cfg_disc = cfg_force_merge(cfg, cfg.DOMAIN_ADAPTATION.STUDENT_CONFIG)

        # Create saving directories
        if resume is False:
            self.create_checkpoint_dir(cfg)

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        assert isinstance(cfg.DOMAIN_ADAPTATION.USE_RESNET, bool)
        model_disc = StudentAccusingDiscriminator(cfg=cfg_disc).cuda()
        opt_disc = self.build_optimizer(cfg_disc, model_disc)

        data_loader = self.build_train_loader(cfg)
        tgt_data_loader = self.build_train_loader(cfg_sup)

        model = create_ddp_model(model, broadcast_buffers=False)
        model_disc = create_ddp_model(model_disc, broadcast_buffers=False)

        self._trainer = DASubTrainer(
            cfg,
            model, model_disc, 
            data_loader, tgt_data_loader,
            optimizer, opt_disc, 
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.scheduler_disc = self.build_lr_scheduler(cfg_disc, opt_disc)
        self.chkpntr = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR+"/model",
            trainer=weakref.proxy(self),
            optimizer=self.optimizer,
        )
        self.chkpntr_disc = DetectionCheckpointer(
            model_disc,
            cfg_disc.OUTPUT_DIR+"/model_disc",
            optimizer=self.opt_disc,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg.clone()
        self.cfg_disc = cfg_disc.clone()
        self.cfg_sup = cfg_sup.clone()

        self.register_hooks(self.build_hooks(resume=resume))
        self.resume_or_load(resume=resume)
    
    def build_hooks(self, resume=False):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            myhooks.LRScheduler(self.optimizer, self.scheduler, verbose=True),
            myhooks.LRScheduler(self.opt_disc, self.scheduler_disc, verbose=False),
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.chkpntr, cfg.SOLVER.CHECKPOINT_PERIOD))
            ret.append(hooks.PeriodicCheckpointer(self.chkpntr_disc, cfg.SOLVER.CHECKPOINT_PERIOD))
            if is_in_colab():
                ret.append(myhooks.ColabModelCopier(ret[-2], resume=resume))
                ret.append(myhooks.ColabModelCopier(ret[-2], resume=resume))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.SOLVER.VERBOSE_PERIOD))

        # if is_in_colab():
        #     ret.append(myhooks.ColabSuspender(print_period=60))

        return ret

    def build_writers(self):
        return [
            myevents.CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def resume_or_load(self, resume=True):
        self.chkpntr = AdetCheckpointer(
            self.model,
            self.cfg.OUTPUT_DIR+"/model",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.chkpntr.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)

        self.chkpntr_disc = AdetCheckpointer(
            self.model_disc,
            self.cfg_disc.OUTPUT_DIR+"/model_disc",
            optimizer=self.opt_disc,
            scheduler=self.scheduler_disc,
        )
        self.chkpntr_disc.resume_or_load(self.cfg_disc.MODEL.WEIGHTS, resume=resume)

        if resume and self.chkpntr.has_checkpoint():
            iteration = self.chkpntr.get_checkpoint_file()[-11:-4]
            iteration = int(iteration) + 1
            self.start_iter = self.iter = iteration

            for _ in range(iteration):
                self.scheduler.step()
                self.scheduler_disc.step()

    def train_loop(self, start_iter:int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()
    
    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """

        mapper = DatasetMapperWithBasis(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg", "uoais"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            # return TextEvaluator(dataset_name, cfg, True, output_folder)
            return
        elif evaluator_type in ["amodal"]:
            if "visible" in cfg.TEST.EVAL_TARGET:
                evaluator_list.append(VisibleEvaluator(dataset_name, output_folder))
            elif "amodal" in cfg.TEST.EVAL_TARGET:
                evaluator_list.append(AmodalEvaluator(dataset_name, output_folder))
            elif "amodal_visible" in cfg.TEST.EVAL_TARGET:
                evaluator_list.append(AmodalVisibleEvaluator(dataset_name, cfg, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
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
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
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
            if "visible_mask" in results_i.keys() or "amodal_mask" in results_i.keys():
                for k in results_i.keys():
                    print("=====>", k)
                    results[dataset_name + k] = results_i[k]
                    if comm.is_main_process():
                        assert isinstance(
                            results_i[k], dict
                        ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                            results_i[k]
                        )
                        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                        print_csv_format(results_i[k])

            else:    
                results[dataset_name] = results_i
                if comm.is_main_process():
                    assert isinstance(
                        results_i, dict
                    ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def create_checkpoint_dir(cls, cfg):
        if os.path.exists(cfg.OUTPUT_DIR) is False:
            os.mkdir(cfg.OUTPUT_DIR)
        if os.path.exists(cfg.OUTPUT_DIR+"/model") is False:
            os.mkdir(cfg.OUTPUT_DIR+"/model")
        if os.path.exists(cfg.OUTPUT_DIR+"/model_disc") is False:
            os.mkdir(cfg.OUTPUT_DIR+"/model_disc")


attr_keys = [
    "model",
    "model_disc",
    "data_loader", 
    "optimizer",
    "opt_disc",
]
for _attr in attr_keys:
    setattr(
        DATrainer,
        _attr,
        property(
            lambda self, x=_attr: getattr(self._trainer, x),
            lambda self, value, x=_attr: setattr(self._trainer, x, value),
        ),
    )
