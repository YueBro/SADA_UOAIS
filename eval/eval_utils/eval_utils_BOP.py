import os
import cv2
import numpy as np
import torch
import glob
import imageio.v2 as imageio
from random import shuffle

# Detectron/adet stuffs
from adet.config import get_cfg
from adet.utils.post_process import detector_postprocess, DefaultPredictor

# Third party
import eval.eval_utils.compute_PRF as compute_PRF
from utils import standardize_image, array_to_tensor, normalize_depth, inpaint_depth
from foreground_segmentation.model import Context_Guided_Network
from termcolor import colored

# My third party
from mycode.tools import masks_to_one
from mycode.tools.dataloaders import dataloader_func

# Special tools
from tqdm import tqdm
from IPython import embed


BACKGROUND_LABEL = 0
BG_LABELS = {}
BG_LABELS["floor"] = [0, 1]
BG_LABELS["table"] = [0, 1, 2]


def eval_visible_on_BOP(args, data_name=None):
    assert (data_name in dataloader_func), "Dataset provided is unknown."

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    H, W = cfg.INPUT.IMG_SIZE

    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # load dataset
    path_datas = dataloader_func[data_name]()
    shuffle(path_datas)
    path_datas = path_datas[:args.number_eval]
    N = len(path_datas)
    print(colored("Evaluation on " + data_name + " dataset: {} rgbs, {} depths, {} visible masks".format(
                N, N, N), "green"))
    
    metrics_all = []
    iou_masks = 0
    num_inst_all = 0 # number of all instances
    num_inst_mat = 0 # number of matched instance

    for i, path_data in enumerate(tqdm(path_datas)):
        # load rgb and depth
        rgb_img = cv2.imread(path_data['color_pth'])
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.imread(path_data['depth_pth'])
        depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        
        # UOAIS-Net inference
        if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)        
        else:
            uoais_input = rgb_img
        # laod GT (annotation) anno: [H, W]
        anno = masks_to_one(path_data['vis_pths'], resize_to=(W, H))
        labels_anno = np.unique(anno)
        labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]
        num_inst_all += len(labels_anno)

        # forward (UOAIS)
        outputs = predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

        if cfg.INPUT.AMODAL:
            pred_masks = instances.pred_visible_masks.detach().cpu().numpy()
        else:
            pred_masks = instances.pred_masks.detach().cpu().numpy()
            
        # CG-Net inference
        if args.use_cgnet:
            fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
            fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
            fg_depth_input = cv2.resize(depth_img, (320, 240)) 
            fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
            fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
            fg_output = fg_model(fg_input.cuda())
            fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
            fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
            fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)
        
            pred_all = np.zeros_like(anno)
            pred = np.zeros_like(anno)
            for i, mask in enumerate(pred_masks):
                iou = np.sum(np.bitwise_and(mask, fg_output)) / np.sum(mask)
                if iou >= 0.5:
                    pred[mask > False] = i+1
                pred_all[mask > False] = i+1
        else: 
            pred = np.zeros_like(anno)
            for i, mask in enumerate(pred_masks):
                pred[mask > False] = i+1
        
        # evaluate
        metrics, assignments = compute_PRF.multilabel_metrics(pred, anno, return_assign=True)
        metrics_all.append(metrics)

        # compute IoU for all instances
        # print(assignments)
        num_inst_mat += len(assignments)
        assign_visible_pred, assign_visible_gt = 0, 0
        assign_visible_overlap = 0
        for gt_id, pred_id in assignments:
            # count area of visible mask (pred & gt)
            mask_pr = pred == pred_id
            mask_gt = anno == gt_id           
            assign_visible_pred += np.count_nonzero(mask_pr)
            assign_visible_gt += np.count_nonzero(mask_gt)
            # count area of overlap btw. pred & gt
            mask_overlap = np.logical_and(mask_pr, mask_gt)
            assign_visible_overlap += np.count_nonzero(mask_overlap)
        if assign_visible_pred+assign_visible_gt-assign_visible_overlap > 0:
            iou = assign_visible_overlap / (assign_visible_pred+assign_visible_gt-assign_visible_overlap)
        else: iou = 0
        iou_masks += iou
    # compute mIoU for all instances
    miou = iou_masks / len(metrics_all)
    
    # sum the values with same keys
    result = {}
    num = len(metrics_all)
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num

    n_sample_str = f"all {len(path_datas)}" if (args.number_eval >= len(path_datas)) else str(args.number_eval)
    print('\n')
    print(colored("Visible Metrics for " + data_name + f" ({n_sample_str} samples)", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
    print('\n')
