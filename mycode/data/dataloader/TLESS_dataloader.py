import os
import json
import numpy as np
import pycocotools.mask as mask_utils

import imageio.v2 as imageio

from detectron2.structures import BoxMode


__all__ = [
    "load_train", "load_val"
]


# def __doc__():

#     obj = {
#         "iscrowd": 0,
#         "bbox": ...,
#         "keypoints": None,
#         "category_id": 0,
#         "segmentation": ...,
#         "visible_mask": ...,
#         "occluded_mask": ...,
#         "occluded_rate": float,
#         "bbox_mode": BoxMode.XYWH_ABS,
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


def load_train(root="datasets/T-LESS", re_calculate=False):
    assert os.path.exists(root)

    # Load pre_calculated "dataset" variable
    pre_calc_json = root + "/TLESS_load_train.json"
    if re_calculate is False:
        if os.path.exists(pre_calc_json):
            print(f"Loading {pre_calc_json}...")
            with open(pre_calc_json, "r") as fp:
                return json.load(fp)
        else:
            print(f"{pre_calc_json} doesn't exist. Re-calculating...")

    dataset = _load(root=root, folder_idxs=range(0,16))
    if re_calculate is False:
        with open(pre_calc_json, "w") as fp:
            json.dump(dataset, fp)
            print(f"Saved to {pre_calc_json}")
    return dataset


def load_val(root="datasets/T-LESS", re_calculate=False):
    assert os.path.exists(root)

    # Load pre_calculated "dataset" variable
    pre_calc_json = root + "/TLESS_load_val.json"
    if re_calculate is False:
        if os.path.exists(pre_calc_json):
            print(f"Loading {pre_calc_json}...")
            with open(pre_calc_json, "r") as fp:
                return json.load(fp)
        else:
            print(f"{pre_calc_json} doesn't exist. Re-calculating...")

    dataset = _load(root=root, folder_idxs=range(16,20))
    if re_calculate is False:
        with open(pre_calc_json, "w") as fp:
            json.dump(dataset, fp)
            print(f"Saved to {pre_calc_json}")
    return dataset


def _load(root, folder_idxs):
    assert os.path.exists(root)

    foldernames = [path for path in os.listdir(root) if os.path.isdir(root+"/"+path)]
    foldernames = [foldernames[idx] for idx in folder_idxs]
    print(foldernames)

    dataset = []
    for folder_idx in range(len(foldernames)):
        foldername = foldernames[folder_idx]
        print(f"{folder_idx}/{len(foldernames)}...", end="\r")

        scene_id = int(foldername)

        with open(root+"/"+foldername+"/"+"scene_gt_info.json", "r") as fp:
            objs_anno = json.load(fp)
        
        for img_id in objs_anno:    # keys are str image id.
            img_rgb_file = f"{root}/{foldername}/rgb/{img_id:0>6s}.png"
            img_depth_file = f"{root}/{foldername}/depth/{img_id:0>6s}.png"
            height = 540
            width = 720

            objs = []
            for obj_id in range(len(objs_anno[img_id])):
                obj_anno = objs_anno[img_id][obj_id]

                vis_mask_file = f"{root}/{foldername}/mask_visib/{img_id:0>6s}_{obj_id:0>6d}.png"
                amo_mask_file = f"{root}/{foldername}/mask/{img_id:0>6s}_{obj_id:0>6d}.png"

                iscrowd = 0
                bbox = obj_anno["bbox_obj"]
                category_id = 0
                segmentation = np.array(imageio.imread(amo_mask_file)>128, dtype="bool", order='F')
                visible_mask = np.array(imageio.imread(vis_mask_file)>128, dtype="bool", order='F')
                occluded_mask = np.logical_xor(segmentation, visible_mask, dtype="bool", order='F')
                occluded_rate = 1 - obj_anno["visib_fract"]
                bbox_mode = BoxMode.XYWH_ABS

                # import matplotlib.pyplot as plt
                # plt.imshow(segmentation)
                # x,y,w,h = bbox
                # plt.plot(
                #     [x,x+w,x+w,x,x],
                #     [y,y,y+h,y+h,y],
                # )
                # plt.savefig("figure.png")
                # exit(0)

                segmentation = mask_utils.encode(segmentation)
                visible_mask = mask_utils.encode(visible_mask)
                occluded_mask = mask_utils.encode(occluded_mask)
                segmentation['counts'] = segmentation['counts'].decode('utf-8')
                visible_mask['counts'] = visible_mask['counts'].decode('utf-8')
                occluded_mask['counts'] = occluded_mask['counts'].decode('utf-8')

                obj = {
                    "iscrowd": iscrowd,
                    "bbox": bbox,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "visible_mask": visible_mask,
                    "occluded_mask": occluded_mask,
                    "occluded_rate": occluded_rate,
                    "bbox_mode": bbox_mode,
                }
                objs.append(obj)
            
            record = {
                "scene_id": scene_id,
                "file_name": img_rgb_file,
                "depth_file_name": img_depth_file,
                "height": height,
                "width": width,
                "image_id": int(img_id),
                "annotations": objs,
            }
            dataset.append(record)
    print(f"{folder_idx+1}/{len(foldernames)}...")
    return dataset


# To re-generate pre-calculated json files, run this script in terminal.
if __name__ == "__main__":
    # _ = load_train(re_calculate=True)
    # _ = load_val(re_calculate=True)
    _load("datasets/T-LESS", range(0,10))
