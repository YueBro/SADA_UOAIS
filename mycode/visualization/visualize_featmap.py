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
from numpy import concatenate as cat
import glob
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

import torch
from detectron2.utils.events import EventStorage

from foreground_segmentation.model import Context_Guided_Network

from mycode.tools import cfg_from_yaml, styler, cfg_force_merge
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
from adet.data.dataset_mapper import DatasetMapperWithBasis
from detectron2.data import build_detection_train_loader

# from IPython import embed


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
        "-n", "--truncation", type=int, default=200,
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


def calculate_pca_and_plot(src_feat, tgt_feat, trunc_n, kernel="linear", title=""):
    if isinstance(src_feat, list):
        src_feat = cat(src_feat)
    if isinstance(tgt_feat, list):
        tgt_feat = cat(tgt_feat)

    idxs = np.arange(src_feat.shape[0])
    np.random.shuffle(idxs)
    src_feat = src_feat[idxs[:trunc_n], ...]

    idxs = np.arange(tgt_feat.shape[0])
    np.random.shuffle(idxs)
    tgt_feat = tgt_feat[idxs[:trunc_n], ...]

    print(src_feat.shape, tgt_feat.shape)

    pca = KernelPCA(n_components=2, kernel=kernel)
    pca.fit(src_feat)
    src_feat = pca.fit_transform(src_feat)
    tgt_feat = pca.fit_transform(tgt_feat)

    plt.scatter(src_feat[:,0], src_feat[:,1], marker='.')
    plt.scatter(tgt_feat[:,0], tgt_feat[:,1], marker='+')
    plt.legend(["Source", "Target"], loc='upper right')
    plt.title(title+f" (kernel: {kernel})")
    plt.grid()

    return src_feat, tgt_feat


if __name__ == "__main__":
    args = get_parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    _seed = random.randint(0, 1000000)

    # Get Configs
    cfg = cfg_from_yaml(args.config_file, do_merge_to_default=True)
    cfg.MODEL.WEIGHTS = glob.glob("output/eval_model/*.pth")[0]
    cfg.SOLVER.IMS_PER_BATCH = 1
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

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
    
    # Build Model
    model = load_model(cfg, "output/eval_model/model_final.pth")
    W, H = cfg.INPUT.IMG_SIZE

    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    cfg_sup = cfg_force_merge(cfg, cfg_sup)
    data_loader_src = iter(build_detection_train_loader(cfg, mapper=DatasetMapperWithBasis(cfg, is_train=False)))
    data_loader_tgt = iter(build_detection_train_loader(cfg_sup, mapper=DatasetMapperWithBasis(cfg_sup, is_train=False)))

    while True:
        roi_src = []
        post_vis_src = []
        post_amo_src = []
        out_vis_src = []
        out_amo_src = []

        roi_tgt = []
        post_vis_tgt = []
        post_amo_tgt = []
        out_vis_tgt = []
        out_amo_tgt = []
        N = args.truncation

        ### SRC ########################################################################################################
        # roi
        n = 0
        while n < N:
            data = next(data_loader_src)
            with EventStorage(0) as storage:
                with torch.no_grad():
                    y, inner_bin = model(data, proposal_append_gt=False)
            roi_feat = inner_bin['roi_feature']
            f_roi = roi_feat.detach().cpu().flatten(1).numpy()
            
            idxs = list(range(f_roi.shape[0]))
            random.shuffle(idxs)
            idxs = idxs[:25]

            f_roi = f_roi[idxs,...]
            n += f_roi.shape[0]
            roi_src.append(f_roi)

        n = 0
        while n < N:
            data = next(data_loader_src)
            with EventStorage(0) as storage:
                with torch.no_grad():
                    y, inner_bin = model(data)
            post_feat = inner_bin['post_feature']      # [N,256,14,14]*3
            output_logits = inner_bin['output_logits']      # keys=[visible, amodal], each with shape [N,1,28,28] (not sigmoid!)

            f_post_vis = post_feat[0].detach().cpu().flatten(1).numpy()
            f_post_amo = post_feat[1].detach().cpu().flatten(1).numpy()
            f_out_vis = output_logits['visible'].detach().cpu().flatten(1).numpy()
            f_out_amo = output_logits['amodal'].detach().cpu().flatten(1).numpy()
            
            idxs = list(range(f_post_vis.shape[0]))
            random.shuffle(idxs)
            idxs = idxs[:25]

            f_post_vis = f_post_vis[idxs,...]
            f_post_amo = f_post_amo[idxs,...]
            f_out_vis = f_out_vis[idxs,...]
            f_out_amo = f_out_amo[idxs,...]

            n += f_post_amo.shape[0]
            post_vis_src.append(f_post_vis)
            post_amo_src.append(f_post_amo)
            out_vis_src.append(f_out_vis)
            out_amo_src.append(f_out_amo)

        ### TGT ########################################################################################################
        # roi
        n = 0
        while n < N:
            data = next(data_loader_tgt)
            with EventStorage(0) as storage:
                with torch.no_grad():
                    y, inner_bin = model(data)
            roi_feat = inner_bin['roi_feature']
            
            idxs = list(range(f_roi.shape[0]))
            random.shuffle(idxs)
            idxs = idxs[:25]

            f_roi = roi_feat.detach().cpu().flatten(1).numpy()[idxs,...]
            n += f_roi.shape[0]
            roi_tgt.append(f_roi)
        
        n = 0
        while n < N:
            data = next(data_loader_tgt)
            with EventStorage(0) as storage:
                with torch.no_grad():
                    y, inner_bin = model(data)
            post_feat = inner_bin['post_feature']      # [N,256,14,14]*3
            output_logits = inner_bin['output_logits']      # keys=[visible, amodal], each with shape [N,1,28,28] (not sigmoid!)

            f_post_vis = post_feat[0].detach().cpu().flatten(1).numpy()
            f_post_amo = post_feat[1].detach().cpu().flatten(1).numpy()
            f_out_vis = output_logits['visible'].detach().cpu().flatten(1).numpy()
            f_out_amo = output_logits['amodal'].detach().cpu().flatten(1).numpy()

            idxs = list(range(f_post_vis.shape[0]))
            random.shuffle(idxs)
            idxs = idxs[:25]

            f_post_vis = f_post_vis[idxs,...]
            f_post_amo = f_post_amo[idxs,...]
            f_out_vis = f_out_vis[idxs,...]
            f_out_amo = f_out_amo[idxs,...]
            n += f_post_amo.shape[0]
            post_vis_tgt.append(f_post_vis)
            post_amo_tgt.append(f_post_amo)
            out_vis_tgt.append(f_out_vis)
            out_amo_tgt.append(f_out_amo)
        
        plt.figure(figsize=[18,3])
        plt.subplot(1,5,1)
        calculate_pca_and_plot(roi_src, roi_tgt, N, kernel="linear", title="$f_\\mathrm{RoI}$")
        plt.xlim(-500,500); plt.ylim(-500,500)
        plt.subplot(1,5,2)
        calculate_pca_and_plot(post_vis_src, post_vis_tgt, N, kernel="rbf", title="$f_\\mathrm{vis}$")
        plt.xlim(-0.6,0.6); plt.ylim(-0.6,0.6)
        plt.subplot(1,5,3)
        calculate_pca_and_plot(post_amo_src, post_amo_tgt, N, kernel="rbf", title="$f_\\mathrm{amo}$")
        plt.xlim(-0.6,0.6); plt.ylim(-0.6,0.6)
        plt.subplot(1,5,4)
        calculate_pca_and_plot(out_vis_src, out_vis_tgt, N, kernel="poly", title="$\\mathcal{V}$")
        plt.xlim(-500,500); plt.ylim(-500,500)
        plt.subplot(1,5,5)
        calculate_pca_and_plot(out_amo_src, out_amo_tgt, N, kernel="poly", title="$\\mathcal{A}$")
        plt.xlim(-500,500); plt.ylim(-500,500)
        plt.show()

        if input("Re-generate (y/n)? >>> ") != 'y':
            break
