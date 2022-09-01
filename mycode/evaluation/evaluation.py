import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import random
import glob

from mycode.evaluation._eval_utils import eval_amodal_occ, eval_visible
from mycode.tools import cfg_from_yaml, styler
from mycode.utils.dataset_config_dict import dataset_config_dict


dataset_name_quick_search_dict = {
    "tless": "TLESS_val",
    "uoais": "uoais_sim_val_amodal",
    "uoais-sim": "uoais_sim_val_amodal",
    "osd": "OSD_val",
    "osd-all": "OSD_all",
}


def get_parser_args():
    parser = argparse.ArgumentParser('UOIS CenterMask')

    # model config   
    parser.add_argument("--config-file", 
        default="./configs/DA_development.yaml", 
        metavar="FILE", help="path to config file")
    parser.add_argument(
        "--dataset-name", type=str, required=True,
    )
    parser.add_argument(
        "-n", "--truncation", type=int, default=None,
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None, help="Random seed for truncation."
    )
    parser.add_argument(
        "--use-cgnet",
        action="store_true",
        help="Use foreground segmentation model to filter our background instances or not"
    )
    parser.add_argument(
        "--cgnet-weight-path",
        type=str,
        default="./foreground_segmentation/rgbd_fg.pth",
        help="path to forground segmentation weight"
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Get Configs
    cfg = cfg_from_yaml(args.config_file, do_merge_to_default=True)
    cfg.MODEL.WEIGHTS = glob.glob("output/eval_model/*.pth")[0]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    # Get proper dataset_name
    dataset_name = dataset_name_quick_search_dict.get(
        args.dataset_name.replace('_','-').lower(), args.dataset_name
    )

    # Update dataset's depth_range into config
    if dataset_config_dict.get(dataset_name, None) is not None:
        cfg_sup = cfg_from_yaml(dataset_config_dict[dataset_name])
        cfg.INPUT.DEPTH_RANGE = cfg_sup.INPUT.DEPTH_RANGE
    
    # Prompts basic infos
    os.system('clear')
    print(f"{styler.bold}YAML path{styler.reset}: \"{args.config_file}\".")
    print(f"{styler.bold}Model path{styler.reset}: \"{cfg.MODEL.WEIGHTS}\".")
    print(f"{styler.bold}Dataset{styler.reset}: \"{dataset_name}\"" + styler.bold + f" ### depth_range: {cfg.INPUT.DEPTH_RANGE}" + styler.reset)
    print(f"{styler.bold}Truncation{styler.reset}: {args.truncation}")

    # Truncation
    if args.truncation is not None:
        if args.seed is None:
            seed = random.randint(0, 2**32)
        else:
            seed = args.seed
        print(f"({styler.bold}Using seed{styler.reset}: {seed})")
        random.seed(seed)
        idxs = list(range(args.truncation))
        random.shuffle(idxs)
    else:
        idxs = None
    
    print()
    eval_amodal_occ(args, dataset_name, cfg=cfg, truncation_idxs=idxs)
    print("="*os.get_terminal_size().columns, end="\n\n")
    eval_visible(args, dataset_name, cfg=cfg, truncation_idxs=idxs)
