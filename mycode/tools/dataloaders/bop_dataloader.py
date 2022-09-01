import os
from copy import deepcopy as dpcpy


class BOP_Dataloader:
    __doc__ = \
    """
    
    """

    def __init__(self, data_path=None):
        self._data_path = data_path

        self._path_datas = None
        self._pixel_datas = None
    
    def get_path_data(self, alternative_path=None):
        """
        return:
            datas = [
                {
                    'color_pth': str,
                    'depth_pth': str,
                    'mask_pths': [str, ...],
                    'vis_pths': [str, ...]
                }, ...
            ]
        """
            
        if (alternative_path == self._data_path) or (alternative_path is None):
            if self._path_datas is not None:
                return dpcpy(self._path_datas)
            path = self._data_path
            self._path_datas = _load_bop_paths_as_dict(data_path=path)
            return self._path_datas
        else:
            path = alternative_path
            return _load_bop_paths_as_dict(data_path=path)


def _load_bop_paths_as_dict(data_path):
    """
    return:
        datas = [
            {
                'color_pth': str,
                'depth_pth': str,
                'mask_pths': [str, ...],
                'vis_pths': [str, ...]
            }, ...
        ]
    """

    assert os.path.exists(data_path), "Error: path invalid."

    datas = []

    for root, foldernames, _ in os.walk(data_path):
        for foldername in foldernames:
            folder_path = root + "/" + foldername
            datas += _load_set(folder_path)
        break
    return datas


def _load_set(folder_path):
    datas = []
    N = 0       # Number of images

    color_folder_path = folder_path + "/" + "rgb"
    depth_folder_path = folder_path + "/" + "depth"
    mask_folder_path = folder_path + "/" + "mask"
    vis_folder_path = folder_path + "/" + "mask_visib"

    assert os.path.exists(color_folder_path), "Error: color path invalid."
    assert os.path.exists(depth_folder_path), "Error: depth path invalid."
    assert os.path.exists(mask_folder_path), "Error: mask path invalid."
    assert os.path.exists(vis_folder_path), "Error: vis-mask path invalid."

    for root, foldernames, filenames in os.walk(color_folder_path):
        assert (filenames!=[]), "Error: no color images found."
        for filename in filenames:
            img_id = int(filename.split(".")[0])
            file_path = root + "/" + filename

            datas.append({
                'img_id': img_id,
                'color_pth': file_path,
                'depth_pth': None,
                'mask_pths': [],
                'vis_pths': []
            })
            N += 1
        break

    datas.sort(key=lambda x: x['img_id'])

    for root, foldernames, filenames in os.walk(depth_folder_path):
        assert (filenames!=[]), "Error: no depth images found."
        for filename in filenames:
            img_id = int(filename.split(".")[0])
            file_path = root + "/" + filename

            datas[img_id]['depth_pth'] = file_path
        break

    for root, foldernames, filenames in os.walk(mask_folder_path):
        assert (filenames!=[]), "Error: no masks found."
        for filename in filenames:
            img_id = int(filename.split("_")[0])
            file_path = root + "/" + filename

            datas[img_id]['mask_pths'].append(file_path)
        break

    for root, foldernames, filenames in os.walk(vis_folder_path):
        assert (filenames!=[]), "Error: no visible masks found."
        for filename in filenames:
            img_id = int(filename.split("_")[0])
            file_path = root + "/" + filename

            datas[img_id]['vis_pths'].append(file_path)
        break

    for data in datas:
        data.pop('img_id')

    return datas