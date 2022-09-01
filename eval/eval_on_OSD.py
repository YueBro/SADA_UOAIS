import warnings
warnings.filterwarnings("ignore")

import argparse
import os

from eval.eval_utils import eval_visible_on_OSD, eval_amodal_occ_on_OSD

from mycode.tools.tools import cfg_from_yaml


if __name__ == "__main__":

    parser = argparse.ArgumentParser('UOIS CenterMask', add_help=False)

    # model config   
    parser.add_argument("--config-file", 
        default="./configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml", 
        metavar="FILE", help="path to config file")    
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument("--vis-only", action="store_true")
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
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./datasets/OSD-0.2-depth",
        help="path to the OSD dataset"
    )


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = cfg_from_yaml(args.config_file, do_merge_to_default=True)
    cfg.MODEL.WEIGHTS = os.path.join("output/eval_model", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    
    print(f"Model path: \"{cfg.MODEL.WEIGHTS}\".")
    print("Dataset: \"OSD\"")
    print("Truncation: \"None\"", end="\n\n")

    if args.vis_only:
        eval_visible_on_OSD(args, cfg)
    else:
        eval_amodal_occ_on_OSD(args, cfg)
        eval_visible_on_OSD(args, cfg)
