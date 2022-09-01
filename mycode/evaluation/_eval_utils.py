# Registration ############################################
import adet.data.builtin            # UOAIS dataset
import mycode.data.registration     # Other datasets
###########################################################

import os
import cv2
import numpy as np
import torch
import random
import imageio.v2 as imageio

# Detectron/adet stuffs
from adet.config import get_cfg
from adet.utils.post_process import detector_postprocess, DefaultPredictor
from detectron2.data.catalog import DatasetCatalog

# Third party
import eval.eval_utils.compute_PRF as compute_PRF
from utils import standardize_image, array_to_tensor, normalize_depth, inpaint_depth
from foreground_segmentation.model import Context_Guided_Network
from termcolor import colored
import pycocotools.mask as mask_tools

# My third party
from mycode.tools import convert_to_rgbd_input, combine_vis_masks

# Special tools
from tqdm import tqdm


BACKGROUND_LABEL = 0
BG_LABELS = {}
BG_LABELS["floor"] = [0, 1]
BG_LABELS["table"] = [0, 1, 2]


def eval_amodal_occ(args, dataset_name, cfg=None, truncation_idxs=None):
    print("--- Amodal evaluation ---")

    if cfg is None:
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.defrost()
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        cfg.freeze()
    
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # load dataset
    dataset = DatasetCatalog.get(dataset_name)
    if truncation_idxs is not None:
        assert isinstance(truncation_idxs, list), "truncation_idxs should be a list of indices"
        truncation_idxs.sort()
        dataset = [dataset[idx] for idx in truncation_idxs]

    print(colored("Evaluation on {} dataset: {} images, {} instances".format(
        dataset_name, len(dataset), sum([len(data['annotations']) for data in dataset])), "green"))
    
    metrics_all = [] # amodal mask evaluation
    num_inst_all_pred = 0 # number of all pred instances
    num_inst_all_gt = 0 # number of all GT instances
    num_inst_occ_pred = 0 # number of occluded prediction
    num_inst_occ_mat = 0 # number of occluded and matched
    num_inst_mat = 0 # number of matched instance
    
    mask_ious, occ_ious = 0, 0
    pre_occ, rec_occ, f_occ = 0, 0, 0
    pre_bou, rec_bou, f_bou = 0, 0, 0
    num_correct = 0 # for occlusion classification
    num_occ_over75 = 0 # number of instance IoU>0.75
    occ_over_75_rate = 0 # rate of instance IoU>0.75

    for i, data in enumerate(tqdm(dataset)):

        # load rgb and depth
        rgb_img = cv2.imread(data['file_name'])
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.imread(data['depth_file_name'])
        depth_img = normalize_depth(depth_img, min_val=cfg.INPUT.DEPTH_RANGE[0], max_val=cfg.INPUT.DEPTH_RANGE[1])
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        
        # # UOAIS-Net inference
        # if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
        #     uoais_input = depth_img
        # elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
        #     uoais_input = np.concatenate([rgb_img, depth_img], -1)
        # else:
        #     uoais_input = rgb_img
        uoais_input = np.concatenate([rgb_img, depth_img], -1)
        # laod GT (amodal masks)
        # img_name = os.path.basename(data['file_name'])[:-4]
        annos = [] # [instance, IMG_H, IMG_W]
        # filtered_amodal_paths = list(filter(lambda p: img_name + "_" in p, amodal_anno_paths))
        # filtered_occlusion_paths = list(filter(lambda p: img_name + "_" in p, occlusion_anno_paths))

        # for anno_path in filtered_amodal_paths:
        #     # get instance id  
        #     inst_id = os.path.basename(anno_path)[:-4].split("_")[-1]
        #     inst_id = int(inst_id)
        #     # load mask image
        #     anno = imageio.imread(anno_path)
        #     anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
        #     # fill mask with instance id
        #     cnd = anno > 0
        #     anno_mask = np.zeros((H, W))
        #     anno_mask[cnd] = inst_id
        #     annos.append(anno_mask)            
        # annos = np.stack(annos)
        # num_inst_all_gt += len(filtered_amodal_paths)

        for inst_id, amo_rle in enumerate([instance['segmentation'] for instance in data['annotations']], start=1):
            # load mask image
            anno = mask_tools.decode(amo_rle)
            anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
            # fill mask with instance id
            cnd = anno > 0
            anno_mask = np.zeros((H, W))
            anno_mask[cnd] = inst_id
            annos.append(anno_mask)            
        annos = np.stack(annos)
        num_inst_all_gt += len(data['annotations'])
        
        # forward (UOAIS)
        outputs = predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

        if not args.use_cgnet:
            pred_masks = instances.pred_masks.detach().cpu().numpy()
            preds = [] # mask per each instance
            for i, mask in enumerate(pred_masks):
                pred = np.zeros((H, W))
                pred[mask > False] = i+1
                preds.append(pred)
                num_inst_all_pred += 1
        else:
            fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
            fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
            fg_depth_input = cv2.resize(depth_img, (320, 240)) 
            fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
            fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
            fg_output = fg_model(fg_input.cuda())
            fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
            fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
            fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

            # filter amodal predictions with foreground mask
            pred_masks = instances.pred_masks.detach().cpu().numpy()
            preds = [] # mask per each instance
            for i, mask in enumerate(pred_masks):
                overlap = np.sum(np.bitwise_and(mask, fg_output)) / np.sum(mask)
                if overlap >= 0.5: # filiter outliers
                    pred = np.zeros((H, W))
                    pred[mask > False] = i+1
                    preds.append(pred)
                    num_inst_all_pred += 1

        if len(preds) > 0:
            preds = np.stack(preds)
        else:
            preds = np.array(preds)
        # evaluate amodal masks
        metrics, assignments = compute_PRF.multilabel_amodal_metrics(preds, annos, return_assign=True)
        metrics_all.append(metrics)

        # count the number of instances
        num_inst_mat += len(assignments)

        if not cfg.INPUT.AMODAL:
            raise ValueError("Cannot support amodal segmentation")

        amodals = instances.pred_masks.detach().cpu().numpy()
        visibles = instances.pred_visible_masks.detach().cpu().numpy()        
        
        # count occluded area of predictions when classified
        all_occ_pred, all_bou_pred = 0, 0
        num_inst_occ_prd_img = 0
        for pred in preds:
            idx = int(pred.max())-1
            amodal = amodals[idx]
            visible = visibles[idx]
            occ = np.bitwise_xor(amodal, visible)
            try:
                cls = instances.pred_occlusions[idx].item()
            except:
                # if area over 5% of amodal mask is not visible
                cls = 1 if np.int64(np.count_nonzero(occ)) / np.int64(np.count_nonzero(amodal)) >= 0.05 else 0
            if not cls: continue
            num_inst_occ_pred += 1
            num_inst_occ_prd_img += 1
            all_occ_pred += np.int64(np.count_nonzero(occ))
            all_bou_pred += np.sum(compute_PRF.seg2bmap(occ))
        # count occluded area of ground truth
        all_occ_gt, all_bou_gt = 0, 0
        occ_rles = [instance['occluded_mask'] for instance in data['annotations'] if instance['occluded_rate']>=0.05]
        for occ_rle in occ_rles:
            occ = mask_tools.decode(occ_rle)
            occ = cv2.resize(occ, (W, H), interpolation=cv2.INTER_NEAREST)
            occ = occ[:,:] > 0
            all_occ_gt += np.int64(np.count_nonzero(occ))
            all_bou_gt += np.sum(compute_PRF.seg2bmap(occ))
            
            # idx = int(os.path.basename(occ_path)[:-4].split("_")[1]) - 1
        # count area with matched instances
        # assign: [[gt_id, pred_id], ... ]
        assign_amodal_pred, assign_visible_pred, assign_amodal_gt = 0, 0, 0
        assign_occ_pred, assign_occ_gt = 0, 0
        assign_amodal_overlap, assign_occ_overlap = 0, 0
        occ_bou_pre, occ_bou_rec = 0, 0
        num_occ_over75_img = 0

        for gt_id, pred_id in assignments:
            ###############
            # AMODAL MASK #
            ###############
            # count area of masks predictions
            amodal = amodals[int(pred_id)-1]
            visible = visibles[int(pred_id)-1]
            assign_amodal_pred += np.count_nonzero(amodal)
            assign_visible_pred += np.count_nonzero(visible)
            # count area of mask GT [ annos: amodal GT ]
            anno = annos[np.where(annos == gt_id)[0][0]] > 0
            assign_amodal_gt += np.count_nonzero(anno)
            # count overlap area of mask btw. pred & GT
            amodal_overlap = np.logical_and(amodal, anno)
            assign_amodal_overlap += np.count_nonzero(amodal_overlap)
                        
            ##################
            # OCCLUSION MASK #
            ##################
            # count area of occlusion prediction
            occ_pred = np.bitwise_xor(amodal, visible)
            assign_occ_pred += np.count_nonzero(occ_pred)

            ############################
            # OCCLUSION CLASSIFICATION #
            ############################
            # count occlusion classification corrects
            # cls_gt = os.path.isfile("{}/occlusion_annotation/{}_{}.png".format(args.dataset_path, img_name, int(gt_id)))
            cls_gt = data['annotations'][int(gt_id)-1]['occluded_rate'] >= 0.05
            try:
                cls_pred = instances.pred_occlusions[int(pred_id)-1].item()
            except:
                cls_pred = 1 if np.int64(np.count_nonzero(occ_pred)) / np.int64(np.count_nonzero(amodal)) >= 0.05 else 0
            num_correct += cls_pred==cls_gt
            if cls_pred==cls_gt and cls_pred == 1:
                num_inst_occ_mat += cls_pred         

            ##################
            # OCCLUSION MASK #
            ##################
            # count area of occlusion GT
            # occ_path = "{}/occlusion_annotation/{}_{}.png".format(args.dataset_path, img_name, int(gt_id))
            # if not os.path.isfile(occ_path) or not cls_pred: continue
            if not data['annotations'][int(gt_id)-1]['occluded_rate']>=0.05 or not cls_pred: continue
            # occ_gt = imageio.imread(occ_path)
            occ_gt = mask_tools.decode(data['annotations'][int(gt_id)-1]['occluded_mask'])
            occ_gt = cv2.resize(occ_gt, (W, H), interpolation=cv2.INTER_NEAREST)
            occ_gt = occ_gt[:,:] > 0
            assign_occ_gt += np.count_nonzero(occ_gt)
            # count overlap area of occlusion btw. pred & GT
            occ_overlap = np.logical_and(occ_pred, occ_gt)
            assign_occ_overlap += np.count_nonzero(occ_overlap)
            
            ############################
            # Over75 of OCCLUSION MASK #
            ############################
            pre = np.int64(np.count_nonzero(occ_overlap)) / np.count_nonzero(occ_pred)
            rec = np.int64(np.count_nonzero(occ_overlap)) / np.count_nonzero(occ_gt)
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            if f > 0.75: 
                num_occ_over75_img += 1
                num_occ_over75 += 1
                        
            ###########################
            # OCCLUSION MASK BOUNDARY #
            ###########################
            pre, rec = compute_PRF.boundary_overlap_occ(gt_id, pred_id, occ_pred, occ_gt)
            occ_bou_pre += pre
            occ_bou_rec += rec
        
        #####################################
        # COMPUTE METRICS in a single image #
        #####################################
        # mIoU of amodal mask (only matched instance)
        if assign_amodal_pred+assign_amodal_gt-assign_amodal_overlap > 0:
            iou = assign_amodal_overlap / (assign_amodal_pred+assign_amodal_gt-assign_amodal_overlap)
        else: iou = 0
        mask_ious += iou
        # mIoU of occlusion mask (only matched instance)
        if assign_occ_pred+assign_occ_gt-assign_occ_overlap > 0:
            iou = assign_occ_overlap / (assign_occ_pred+assign_occ_gt-assign_occ_overlap)
        else: iou = 0
        occ_ious +=iou
        
        # number of occluded instances in one image
        num_pred = num_inst_occ_prd_img
        # num_gt = len(filtered_occlusion_paths)
        num_gt = len(occ_rles)
        if num_pred == 0 and num_gt > 0:
            pre_occ += 1
            pre_bou += 1
        elif num_pred > 0 and num_gt == 0:
            rec_occ += 1
            rec_bou += 1
        elif num_pred == 0 and num_gt == 0:
            pre_occ += 1
            rec_occ += 1
            f_occ += 1
            pre_bou += 1
            rec_bou += 1
            f_bou += 1
            occ_over_75_rate += 1
        else:
            assert (num_pred > 0) and (num_gt > 0)
            # P, R, F of occlusion mask (all instance)
            pre = assign_occ_overlap / all_occ_pred if all_occ_pred > 0 else 0
            rec = assign_occ_overlap / all_occ_gt if all_occ_gt > 0 else 0
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            pre_occ += pre
            rec_occ += rec
            f_occ += f
            # P, R, F of occlusion boundary (all instance)
            pre = occ_bou_pre / all_bou_pred if all_bou_pred > 0 else 0
            rec = occ_bou_rec / all_bou_gt if all_bou_gt > 0 else 0
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            pre_bou += pre
            rec_bou += rec
            f_bou += f
            occ_over_75_rate += num_occ_over75_img / num_gt

    ###############################################
    # get average metirc values among test images #
    ###############################################
    num = len(metrics_all)
    mask_ious /= num 
    occ_ious /= num 
    pre_occ /= num 
    rec_occ /= num 
    f_occ /= num 
    pre_bou /= num 
    rec_bou /= num 
    f_bou /= num 
    occ_over_75_rate /= num

    occ_cls_acc = num_correct / num_inst_mat * 100 if num_inst_mat > 0 else 0
    occ_cls_pre = num_correct / num_inst_all_pred * 100 if num_inst_all_pred > 0 else 0
    occ_cls_rec = num_correct / num_inst_all_gt * 100 if num_inst_all_gt > 0 else 0
    occ_cls_f = (2*occ_cls_pre*occ_cls_rec) / (occ_cls_pre+occ_cls_rec) if occ_cls_pre+occ_cls_rec > 0 else 0

    # sum the values with same keys
    result = {}
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num

    print('\n')
    print(colored("Amodal Metrics on {} (for {} instances)".format(dataset_name, num_inst_all_gt), "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, mask_ious
    ))
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
    print('\n')
    print(colored("Occlusion Metrics on {}".format(dataset_name), "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        pre_occ*100, rec_occ*100, f_occ*100, 
        pre_bou*100, rec_bou*100, f_bou*100,
        occ_over_75_rate*100, occ_ious        
    ))
    print(colored("---------------------------------------------", "green"))
    print('\n')
    print(colored("Occlusion Classification on {}".format(dataset_name), "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("  P   R   F   ACC")
    print("{:.1f} {:.1f} {:.1f} {:.1f}".format(
        occ_cls_pre, occ_cls_rec, occ_cls_f, occ_cls_acc        
    ))
    print(colored("---------------------------------------------", "green"))
    print('\n')

    return



def eval_visible(args, dataset_name, cfg=None, truncation_idxs=None):
    print("--- Visible evaluation ---")
    
    if cfg is None:
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.defrost()
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        cfg.freeze()
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # Load dataset
    dataset = DatasetCatalog.get(dataset_name)
    if truncation_idxs is not None:
        assert isinstance(truncation_idxs, list), "truncation_idxs should be a list of indices"
        truncation_idxs.sort()
        dataset = [dataset[idx] for idx in truncation_idxs]

    print(colored("Evaluation on {} dataset: {} images, {} instances".format(
        dataset_name, len(dataset), sum([len(data['annotations']) for data in dataset])), "green"))
    
    metrics_all = []
    iou_masks = 0
    num_inst_all = 0 # number of all instances
    num_inst_mat = 0 # number of matched instance

    for i, (data) in enumerate(tqdm(dataset)):

        # load rgb and depth
        rgb_img = cv2.imread(data['file_name'])
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.imread(data['depth_file_name'])
        depth_img = normalize_depth(depth_img, min_val=cfg.INPUT.DEPTH_RANGE[0], max_val=cfg.INPUT.DEPTH_RANGE[1])
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
        anno = combine_vis_masks([instance['visible_mask'] for instance in data['annotations']])
        anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
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

    print('\n')
    print(colored("Visible Metrics for {}".format(dataset_name), "green", attrs=["bold"]))
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
