import os
from copy import deepcopy as dpcpy


class OSD_Dataloader:
    __doc__ = \
    """
    
    """

    def __init__(self, data_path='datasets/OSD-0.2-depth'):
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
                    'amodal_pths': [str, ...],
                    'vis_pths': str,
                    'occ_anno': [str, ...],
                }, ...
            ]
        """
        if (alternative_path==self._data_path or alternative_path is None) and (self._path_datas is not None):
            return dpcpy(self._path_datas)

        _path_datas = {}

        for root, foldernames, filenames in os.walk(alternative_path + "/image_color"):
            for filename in filenames:
                img_key = filename.split(".")[0]
                path = root + "/" + filename
                _path_datas[img_key] = {
                    'color_pth': path,
                    'depth_pth': None,
                    'amodal_pths': {},
                    'vis_pths': None,
                    'occ_anno': {},
                }
        
        assert len(_path_datas) > 0, "Path error."

        for root, foldernames, filenames in os.walk(alternative_path + "/disparity"):
            for filename in filenames:
                img_key = filename.split(".")[0]
                path = root + "/" + filename
                _path_datas[img_key]['depth_pth'] = path

        for root, foldernames, filenames in os.walk(alternative_path + "/annotation"):
            for filename in filenames:
                img_key = filename.split(".")[0]
                path = root + "/" + filename
                _path_datas[img_key]['vis_pths'] = path

        for root, foldernames, filenames in os.walk(alternative_path + "/amodal_annotation"):
            for filename in filenames:
                img_key = filename.split("_")[0]
                idx = int(filename.split("_")[1].split(".")[0])
                path = root + "/" + filename
                _path_datas[img_key]['amodal_pths'][idx] = filename
            for img_key in _path_datas:
                _path_datas[img_key]['amodal_pths'] = [element[1] for element in sorted(_path_datas[img_key]['amodal_pths'].items())]

        for root, foldernames, filenames in os.walk(alternative_path + "/occlusion_annotation"):
            for filename in filenames:
                img_key = filename.split("_")[0]
                idx = int(filename.split("_")[1].split(".")[0])
                path = root + "/" + filename
                _path_datas[img_key]['occ_anno'][idx] = filename
            for img_key in _path_datas:
                _path_datas[img_key]['occ_anno'] = [element[1] for element in sorted(_path_datas[img_key]['occ_anno'].items())]
        
        _path_datas = [_path_datas[key] for key in _path_datas]

        if (alternative_path == self._data_path) or (alternative_path is None):
            self._path_datas = _path_datas
        
        return dpcpy(_path_datas)

