# Registration ############################################
import adet.data.builtin            # UOAIS dataset
import mycode.data.registration     # Other datasets
###########################################################

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import random
import numpy as np
import glob

import torch
from adet.utils.post_process import DefaultPredictor
from detectron2.data.catalog import DatasetCatalog

from foreground_segmentation.model import Context_Guided_Network

from mycode.tools import cfg_from_yaml, styler
from mycode.tools import (
    convert_to_rgbd_input,
    filter_out_background_with_fg_model,
    mark_result_on_img,
    cv2imshow,
    load_model,
)
from mycode.utils import get_no_obj_im
from mycode.utils.dataset_config_dict import dataset_config_dict
from mycode.utils import (
    keyLEFT,
    keyRIGHT,
    keyESC
)


dataset_name_quick_search_dict = {
    "tless": "TLESS_val",
    "uoais": "uoais_sim_val_amodal",
    "uoais-sim": "uoais_sim_val_amodal",
    "osd": "OSD_val",
    "osd-all": "OSD_all",
}

no_obj_im = get_no_obj_im((640, 480))


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
        "-n", "--truncation", type=int, default=20,
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

    # Set random seed
    if args.seed is None:
        seed = random.randint(0, 2**32)
    else:
        seed = args.seed
    print(f"({styler.bold}Using seed{styler.reset}: {seed})")
    random.seed(seed)
    
    # Build Model
    model = load_model(cfg, "output/eval_model/model_final.pth")
    model.eval()
    W, H = cfg.INPUT.IMG_SIZE

    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()
    
    # Get dataset and truncate
    dataset = DatasetCatalog.get(dataset_name)
    random.shuffle(dataset)
    dataset = dataset[:args.truncation]
    
    IMG_I = len(dataset)
    img_i = 0
    result_cache = [None] * IMG_I
    while True:
        if result_cache[img_i] is None:
            x, bgr_im, dep_img = convert_to_rgbd_input(
                dataset[img_i]['file_name'],
                dataset[img_i]['depth_file_name'],
                img_size=(W, H),
                return_inter_imgs=True,
                depth_range=cfg.INPUT.DEPTH_RANGE,
            )

            # Get result from model
            x = {"image": x, "height": 480, "width": 640}
            y = model([x])[0]["instances"]

            # Filter out background
            if args.use_cgnet is True:
                y = filter_out_background_with_fg_model(fg_model, bgr_im, dep_img, y, print_idxs=True)

            # Show image
            num_object = len(y)
            if num_object == 0:
                print(f"{styler.bold_blue}No object detected!{styler.reset}")
                img = np.hstack([bgr_im, dep_img, no_obj_im])
            else:
                img = mark_result_on_img(bgr_im, y, depth_img=dep_img, do_interp_depth=True)
            result_cache[img_i] = img
        else:
            img = result_cache[img_i]

        allowed_keys = \
            [keyRIGHT, keyESC] if img_i == 0 else \
            [keyLEFT, keyESC] if img_i == IMG_I-1 else \
            [keyLEFT, keyRIGHT, keyESC]
        key_press = cv2imshow(
            img, 
            title=f"Press ESC, LEFT, and RIGHT to interact {img_i+1}/{IMG_I}",
            keys_allowed=allowed_keys
        )
        if key_press == keyLEFT:
            img_i = max([0, img_i-1])
        elif key_press == keyRIGHT:
            img_i = min([IMG_I-1, img_i+1])
        elif key_press == keyESC:
            break
