__notes__ = "="*40 + """
Yulin Shen's Notes:
    This script is used to train model.
""" + "="*40

import warnings
warnings.filterwarnings("ignore")

# Registrations ####################################################################
import mycode.data.registration             # Datasets
import adet.modeling.domain_shift_modules   # meta_arch
####################################################################################

import os
import torch
import shutil

import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, launch
from detectron2.utils.logger import setup_logger

from mycode.tools.det2_adet_subs import cfg_from_yaml
from mycode.tools import is_in_colab

##### Selections of trainer #####
from mycode.tools.det2_adet_subs.trainers_SADA import DATrainer


def setup(args):
    cfg = cfg_from_yaml(args.config_file, do_merge_to_default=True)
    cfg.freeze()

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = DATrainer(cfg, resume=args.resume)
    if is_in_colab():
        folder_name = cfg.OUTPUT_DIR.split('/')[-1]
        shutil.copyfile(args.config_file, f"../drive/MyDrive/train_result/{folder_name}/cfg.yaml")
    train_result = trainer.train()
    return train_result


if __name__ == "__main__":
    print(__notes__)
    
    parser = default_argument_parser()
    parser.add_argument("--config-file-sup", type=str, default="", help="Configure file for target domain dataset")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    args = parser.parse_args()

    if args.config_file == "":
        args.config_file = "configs/DA_SADA.yaml"
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    # main(args)
