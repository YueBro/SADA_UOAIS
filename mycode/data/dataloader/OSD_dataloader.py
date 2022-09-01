import os
import json
import numpy as np
import pycocotools.mask as mask_utils

import imageio.v2 as imageio

from detectron2.structures import BoxMode

from IPython import embed


__all__ = [
    "load_train", "load_val"
]


# def __doc__():

#     obj = {
#         "iscrowd": 0,
#         "bbox": ...,
#         "keypoints": None,
#         "category_id": 1,
#         "segmentation": ...,
#         "visible_mask": ...,
#         "occluded_mask": ...,
#         "occluded_rate": float,
#         "bbox_mode": BoxMode.XYWH_ABS,
#         "category_id": 0,
#     }

#     objs = [obj, ...]

#     record = {
#         "file_name": str,
#         "depth_file_name": str,
#         "height": ...,
#         "weight": ...,
#         "image_id": ...,
#         "annotations": objs,
#     }

#     dataset = [record, ...]

#     return dataset


def load_train(root="datasets/OSD-0.2-depth", re_calculate=False):
    assert os.path.exists(root)

    pre_calc_json = root + "/OSD_load_train.json"
    if (re_calculate is True) or (os.path.exists(pre_calc_json) is False):
        print("Re-generating OSD load json file...")
        _load(root)

    with open(pre_calc_json, 'r') as fp:
        print("Loading OSD-train json file...")
        dataset = json.load(fp)
    return dataset


def load_val(root="datasets/OSD-0.2-depth", re_calculate=False):
    assert os.path.exists(root)

    pre_calc_json = root + "/OSD_load_val.json"
    if (re_calculate is True) or (os.path.exists(pre_calc_json) is False):
        print("Re-generating OSD load json file...")
        _load(root)

    with open(pre_calc_json, 'r') as fp:
        print("Loading OSD-val json file...")
        dataset = json.load(fp)
    return dataset


def load_all(root="datasets/OSD-0.2-depth", re_calculate=False):
    assert os.path.exists(root)

    pre_calc_json = root + "/OSD_load_all.json"
    if (re_calculate is True) or (os.path.exists(pre_calc_json) is False):
        print("Re-generating OSD load json file...")
        _load(root)

    with open(pre_calc_json, 'r') as fp:
        print("Loading OSD-all json file...")
        dataset = json.load(fp)
    return dataset


def _load(root):
    assert os.path.exists(root)

    rgb_folder = root + "/image_color"
    dep_folder = root + "/disparity"
    amo_folder = root + "/amodal_annotation"
    occ_folder = root + "/occlusion_annotation"

    dataset = []
    img_idx = 0
    name_to_id_dict = {}
    for _, _, filenames in os.walk(rgb_folder):
        for filename in filenames:
            name_to_id_dict[filename.split('.')[0]] = img_idx
            dataset.append({
                "file_name": rgb_folder + "/" + filename,
                "depth_file_name": dep_folder + "/" + filename,
                "height": 480,
                "width": 640,
                "image_id": img_idx,
                "annotations": [],
            })
            img_idx += 1

    for _, _, filenames in os.walk(amo_folder):
        for filename in filenames:
            img_name = filename.split('_')[0]
            img_idx = name_to_id_dict[img_name]

            segmentation = imageio.imread(amo_folder + '/' + filename) > 0
            bbox = _calculate_bounding_box(segmentation)

            if os.path.exists(occ_folder + '/' + filename) is True:
                occluded_mask = imageio.imread(occ_folder + '/' + filename) > 0
                visible_mask = np.bitwise_xor(segmentation, occluded_mask)
                occluded_rate = np.sum(occluded_mask) / np.sum(segmentation)
                # print(occluded_rate)
                # from mycode.tools import cv2imshow
                # cv2imshow(np.concatenate([segmentation, occluded_mask, visible_mask],-1) * 100)
            else:
                visible_mask = np.array(segmentation)
                occluded_mask = np.zeros_like(segmentation)
                occluded_rate = 0.0
            
            # import matplotlib.pyplot as plt
            # plt.imshow(segmentation)
            # x,y,w,h = bbox
            # plt.plot(
            #     [x,x+w,x+w,x,x],
            #     [y,y,y+h,y+h,y],
            # )
            # plt.savefig("figure.png")
            # exit(0)

            segmentation = mask_utils.encode(np.array(segmentation, order='F'))
            visible_mask = mask_utils.encode(np.array(visible_mask, order='F'))
            occluded_mask = mask_utils.encode(np.array(occluded_mask, order='F'))
            segmentation['counts'] = segmentation['counts'].decode('utf-8')
            visible_mask['counts'] = visible_mask['counts'].decode('utf-8')
            occluded_mask['counts'] = occluded_mask['counts'].decode('utf-8')

            dataset[img_idx]["annotations"].append({
                "iscrowd": 0,
                "bbox": bbox,
                "category_id": 0,
                "segmentation": segmentation,
                "visible_mask": visible_mask,
                "occluded_mask": occluded_mask,
                "occluded_rate": occluded_rate,
                "bbox_mode": BoxMode.XYWH_ABS,
            })

    dataset_train = dataset[:40] + dataset[40::2]
    dataset_val = dataset[40::2]
    with open(root+'/'+'OSD_load_train.json', 'w') as fp:
        json.dump(dataset_train, fp)
    with open(root+'/'+'OSD_load_val.json', 'w') as fp:
        json.dump(dataset_val, fp)
    with open(root+'/'+'OSD_load_all.json', 'w') as fp:
        json.dump(dataset, fp)

    return dataset


def _calculate_bounding_box(mask: np.ndarray):
    assert mask.ndim == 2, "Must be 2 dimension."

    mask = mask > 0
    Y, X = mask.shape

    xmin = xmax = ymin = ymax = None
    for x in range(0, X):
        if np.sum(mask[:,x]) > 0:
            xmin = x
            break
    for x in range(X-1, -1, -1):
        if np.sum(mask[:,x]) > 0:
            xmax = x
            break
    for y in range(0, Y):
        if np.sum(mask[y,:]) > 0:
            ymin = y
            break
    for y in range(Y-1, -1, -1):
        if np.sum(mask[y,:]) > 0:
            ymax = y
            break
    
    bbox = [
        xmin,
        ymin,
        xmax-xmin,
        ymax-ymin,
    ]
    return bbox
    

# To re-generate pre-calculated json files, run this script in terminal.
if __name__ == "__main__":
    # _ = load_train(re_calculate=True)
    # _ = load_val(re_calculate=True)
    pass
