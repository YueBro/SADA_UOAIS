# Standard imports
import os
import torch
import cv2
import glob
import numpy as np
from copy import deepcopy as dpcpy
import json
import time
import pickle as pkl

# Third party imports
import imageio.v2 as imageio
from .det2_adet_subs.config import cfg_force_merge, cfg_from_yaml
from utils import inpaint_depth
from adet.utils.visualizer import visualize_pred_amoda_occ
import pycocotools.mask as mask_tools
from foreground_segmentation.model import Context_Guided_Network
from utils import *

# Classes or methods in adet and Detectron2 
from adet.config import get_cfg as get_default_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from adet.utils.post_process import detector_postprocess

# Types / Objects
from typing import Sequence
from detectron2.structures import Instances, Boxes


__all__ = [
    "is_in_colab", "load_model", "load_cgnet_model", "filter_out_background_with_fg_model", "get_cfg_for_eval",
    "get_img_paths", "get_img_paths", "convert_to_rgbd_input", "convert_BGR_img", "cv2imshow", "mark_result_on_img",
    "rle_to_np_mask", "read_UOAIS_visible_mask", "masks_to_one", "IterRecorder", "get_time_str",
    "time_seconds_to_str", "combine_vis_masks"
]


def is_in_colab():
    # raise NotImplementedError("Function under bug fixing. Please don't use this function.")
    if os.path.exists("../drive/MyDrive"):
        return True
    else:
        return False


def load_model(cfg, model_pth=None):
    model = build_model(cfg)
    model_pth = "output/eval_model/model_final.pth" if (model_pth is None) else cfg.MODEL.WEIGHTS
    print(f"Using checkpoint: {model_pth}")
    DetectionCheckpointer(model).load(model_pth)
    return model


def load_cgnet_model(weight_path='foreground_segmentation/rgbd_fg.pth'):
    checkpoint = torch.load(weight_path)
    fg_model = Context_Guided_Network(classes=2, in_channel=4)
    fg_model.load_state_dict(checkpoint['model'])
    fg_model.cuda()
    return fg_model


def filter_out_background_with_fg_model(fg_model, bgr_img, dep_img_inpaint, pred_instances, print_idxs=False):
    H, W = bgr_img.shape[:2]

    fg_rgb_input = standardize_image(cv2.resize(bgr_img, (320, 240)))
    fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
    fg_depth_input = cv2.resize(dep_img_inpaint, (320, 240))
    fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
    fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
    fg_output = fg_model(fg_input.cuda())
    fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
    fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
    fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

    pred_instances = detector_postprocess(pred_instances, H, W).to('cpu')

    preds = pred_instances.pred_masks.detach().cpu().numpy()
    pred_visibles = pred_instances.pred_visible_masks.detach().cpu().numpy()
    bboxes = pred_instances.pred_boxes.tensor.detach().cpu().numpy()
    pred_occs = pred_instances.pred_occlusions.detach().cpu().numpy()

    remove_idxs = []
    for i, pred_visible in enumerate(pred_visibles):
        iou = np.sum(np.bitwise_and(pred_visible, fg_output)) / np.sum(pred_visible)
        if iou < 0.5: 
            remove_idxs.append(i)
    if print_idxs is True:
        print(f"Removing {remove_idxs}")
    preds = np.delete(preds, remove_idxs, 0)
    pred_visibles = np.delete(pred_visibles, remove_idxs, 0)
    bboxes = np.delete(bboxes, remove_idxs, 0)
    pred_occs = np.delete(pred_occs, remove_idxs, 0)

    new_instances = Instances(
        (H, W),
        pred_masks=torch.tensor(preds),
        pred_visible_masks=torch.tensor(pred_visibles),
        pred_boxes=Boxes(bboxes),
        pred_occlusions=torch.tensor(pred_occs),
    )

    return new_instances


def get_cfg_for_eval(
    config_pth="configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml",
):
    cfg = cfg_from_yaml(config_pth, do_merge_to_default=True)
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "model_final.pth")):
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    elif os.path.exists(os.path.join(cfg.OUTPUT_DIR, "model", "model_final.pth")):
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model", "model_final.pth")
    else:
        print(f"Warning: Model weight not found in \"{cfg.OUTPUT_DIR}\"!")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    return cfg


def get_img_paths(*paths, keywords=None, image_format="png"):
    if keywords is not None:
        assert len(keywords) == len(paths), "len(paths) not matching len(keywords)."
        assert len(keywords) == len(set(keywords)), "Keywords shouldn't have duplicates."

    sub_paths = []
    for path in paths:
        sub_paths.append( sorted(glob.glob(("{}/*."+image_format).format(path))) )
    
    if keywords is None:
        return sub_paths
    else:
        return dict(zip(keywords, sub_paths))


def convert_to_rgbd_input(
    bgr_img,
    depth_img,
    img_size=None,
    return_inter_imgs=False,
    do_transpose=True,
    to_tensor=True,
    depth_range=[2500, 15000],
):
    depth_min, depth_max = depth_range

    # Load RGB and resize
    if isinstance(bgr_img, str):            # "bgr_img" is a path
        _bgr_img = cv2.imread(bgr_img)
        assert (_bgr_img is not None), f"File \"{bgr_img}\" doesn't exist."
        bgr_img = _bgr_img
    if img_size is not None:
        W, H = img_size
    else:
        H, W = bgr_img.shape[:2]
    bgr_img = cv2.resize(bgr_img, (W, H))

    # Load depth, resize, and inpaint
    if isinstance(depth_img, str):          # "depth_img" is a path
        depth_img = imageio.imread(depth_img)
    depth_img = depth_img.astype(np.float32)
    depth_img[depth_img > depth_max] = depth_max
    depth_img[depth_img < depth_min] = depth_min
    depth_img = (depth_img - depth_min) / (depth_max - depth_min) * 255
    depth_img = np.expand_dims(depth_img, -1)
    depth_img = np.uint8(np.repeat(depth_img, 3, -1))
    depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_img = inpaint_depth(depth_img)

    # Concatenate
    img = np.concatenate([bgr_img, depth_img], -1)

    # Make float32, (W,H,6) to (6,W,H), wrap as tensor.
    img = img.astype("float32")
    if do_transpose is True:
        img = img.transpose(2, 0, 1)
    if to_tensor is True:
        img = torch.as_tensor(img)

    if return_inter_imgs is False:
        return img
    else:
        return img, bgr_img, depth_img


def convert_BGR_img(img: Sequence, to="rgb"):
    im = np.array(img).transpose(1,2,0)

    if to.lower()=="rgb":
        im = im[:,:,:-4:-1]
    elif to.lower()=="bgr":
        im = im[:,:,-3:]

    im = im.astype(np.int32)

    return im


def cv2imshow(img, title="Press ANY key to quit", do_interp=False, keys_allowed=None):
    im = np.array(img, dtype=np.uint8)
    if im.ndim==2:
        im = np.repeat(im[:,:,None],3,-1)
    else:
        assert im.ndim==3 and im.shape[2]==3, f"cv2imshow() cannot have input with shape \"{tuple(im.shape)}\""
    if do_interp is True:
        im = np.interp(im.astype(np.float32), (np.min(im), np.max(im)), (0, 255)).astype(np.uint8)

    # If working in locally...
    # if "google.colab" not in sys.modules:
    if is_in_colab() is False:
        cv2.imshow(title, im)
        while True:
            key = cv2.waitKey(0)
            if (keys_allowed is not None) and (key in keys_allowed):
                break
        cv2.destroyAllWindows()
    
    # If working in colab...
    else:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        key_dict = {
            "left": 81,
            "up": 82,
            "right": 83,
            "down": 84,
            "esc": 27
        }
        clear_output()
        plt.imshow(im[:,:,::-1])
        plt.axis('off')
        plt.show()
        time.sleep(1)
        prompt = "Your key entry >>> "
        while True:
            key = input(prompt).lower()
            if key in key_dict:
                key = key_dict[key]
                break
            elif len(key)==1:
                key = ord(key)
                break
            else:
                prompt = "Invalid, please re-enter >>> "
    return key


def mark_result_on_img(bgr_img, instance, depth_img=None, do_interp_depth=False, return_pred_only=False):
    preds = instance.pred_masks.detach().cpu().numpy()
    bboxes = instance.pred_boxes.tensor.detach().cpu().numpy()
    pred_occs = instance.pred_occlusions.detach().cpu().numpy()
    
    idx_shuf = np.concatenate((np.where(pred_occs==1)[0] , np.where(pred_occs==0)[0])) 
    preds, pred_occs, bboxes = preds[idx_shuf], pred_occs[idx_shuf], bboxes[idx_shuf]
    vis_img = visualize_pred_amoda_occ(bgr_img, preds, bboxes, pred_occs)

    if return_pred_only is True:
        return dpcpy(vis_img)

    if depth_img is not None:
        if do_interp_depth is True:
            depth_img = np.interp(depth_img, [np.min(depth_img), np.max(depth_img)], [0,255]).astype(bgr_img.dtype)
        vis_all_img = np.hstack([bgr_img, depth_img, vis_img])
    else:
        vis_all_img = np.hstack([bgr_img, vis_img])

    return dpcpy(vis_all_img)


def rle_to_np_mask(rle_string, sizes, dtype=None):
    if isinstance(rle_string, str):
        rle_string = str.encode(rle_string)
    
    assert isinstance(rle_string, bytes), "\"rle_string\" must be str or bytes."

    mask = mask_tools.decode({'size': sizes, 'counts': rle_string})
    if dtype is None:
        return mask
    else:
        return mask.astype(dtype)


def read_UOAIS_visible_mask(coco_path, dtype=None, n_truncate=None):
    """
        n_truncate: smaller number of annotations to return.
    """
    with open(coco_path, "r") as fp:
        coco_anno = json.load(fp)
    
    bgr_paths = []
    dep_paths = []
    vis_masks = []
    
    res_id = -1
    I = len(coco_anno['annotations'])
    I = I if (n_truncate is None) else min([I, n_truncate])
    item_value = None
    for i in range(I):      # Assume image_id increases 0 or 1 in each iter.
        img_id = coco_anno['annotations'][i]['image_id']
        if img_id == res_id+1:
            res_id += 1
            item_value = 1
            bgr_paths.append("datasets/UOAIS-Sim/val/" + coco_anno['images'][i]['file_name'])
            dep_paths.append("datasets/UOAIS-Sim/val/" + coco_anno['images'][i]['depth_file_name'])
            vis_masks.append(np.zeros(shape=(480,640), dtype=np.uint8))

        vis_sizes = coco_anno['annotations'][i]['visible_mask']['size']
        vis_rle = coco_anno['annotations'][i]['visible_mask']['counts']
        vis_mask = rle_to_np_mask(vis_rle, vis_sizes)
        vis_masks[-1] += vis_mask*item_value

        item_value += 1

    if dtype is not None:
        vis_masks = [mask.astype(dtype) for mask in vis_masks]

    return bgr_paths, dep_paths, vis_masks


def combine_vis_masks(masks):
    """
    Takes in a list of masks, where each mask can be "rle" or "ndarray" format.
    Outputs the combined mask for visible mask representation.
    """

    # Convert to ndarray
    if isinstance(masks[0], dict):
        masks = [mask_tools.decode(masks[i])>0 for i in range(len(masks))]
    else:
        masks = [(masks[i] > 0) for i in range(len(masks))]

    item_value = 1
    mask_comb = np.zeros(shape=masks[0].shape, dtype=np.int64)
    for mask in masks:
        mask_comb = mask_comb * ~mask + mask*item_value
        # mask_comb += mask*item_value
        item_value += 1
    return mask_comb.astype(np.uint8)


def masks_to_one(*masks, dtype=np.uint8, resize_to=None):
    if len(masks) == 1:
        masks = masks[0]

    if isinstance(masks[0], str):
        all_mask = imageio.imread(masks[0])
    else:
        all_mask = (np.array(masks[0]) > 0).astype(dtype)

    for value, mask in enumerate(masks[1:], start=2):
        if isinstance(mask, str):
            mask = imageio.imread(mask)
        all_mask *= ~(mask > 0)
        all_mask += value * (mask > 0).astype(dtype)

    if resize_to is not None:
        W, H = resize_to
        all_mask = cv2.resize(all_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return all_mask


class IterRecorder:
    """
    Records values produced in each iteration. 
    """
    def __init__(self, *keys, load_path=None):
        if load_path is not None:
            self.load(load_path)
        else:
            if len(keys) == 0:
                self.values = []
                self.length = 0
                self.keys = None
            else:
                if len(keys) == 1:
                    keys = keys[0]
                self.values = [[] for _ in range(len(keys))]
                self.length = len(keys)
                self.keys = list(keys)
    
    def append(self, *values):
        if len(values)==1:
            values = values[0]
        if self.length == 0:
            self.length = len(values)
            for value in values:
                self.values.append([value])
        else:
            assert len(values)==self.length, "Number of values and pre-defined length must agree!"
            for i in range(self.length):
                self.values[i].append(values[i])
    
    def __getitem__(self, val):
        if isinstance(val, int):
            return [self.values[i][val] for i in range(len(self.values))]
        elif len(val) == 2:
            return self.values[val[1]][val[0]]
    
    def set_keys(self, *keys):
        if len(keys)==1:
            keys = keys[0]
        if self.length != 0:
            assert len(keys)==self.length, "Number of keys and pre-defined length must agree!"
            self.keys = keys
        else:
            self.__init__(keys)

    def return_dict(self):
        if self.keys is not None:
            return dict(zip(self.keys, self.values))
        else:
            return dict(zip(range(1, len(self.values)+1), self.values))
    
    def save(self, path):
        values = self.return_dict()
        with open(path, "wb") as fp:
            pkl.dump(values, fp)
    
    def load(self, path):
        with open(path, "rb") as fp:
            values = pkl.load(fp)
        
        self.__init__(list(values))
        for i in range(len(list(values.items())[0][1])):
            value_list = []
            for key in values.keys():
                value_list.append(values[key][i])
            self.append(value_list)


def get_time_str():
    t = time.localtime()
    return time.strftime("%H:%M:%S", t)


def time_seconds_to_str(t):
    t = int(t)
    return f"{t//3600:0>2}:{(t%3600)//60:0>2}:{t%3600%60:0>2}"

