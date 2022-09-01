import os
from copy import deepcopy as dpcpy

from mycode.tools import get_img_paths


class OCID_Dataloader:
    __doc__ = \
    """
    
    """

    def __init__(self, data_path='datasets/OCID-dataset'):
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
                }, ...
            ]
        """
        if (alternative_path==self._data_path or alternative_path is None) and (self._path_datas is not None):
            return dpcpy(self._path_datas)

        _path_datas = []

        for root, foldernames, filenames in os.walk(alternative_path):
            if ("depth" in foldernames) and ("rgb" in foldernames):
                paths = get_img_paths(
                    root + "/rgb",
                    root + "/depth",
                    keywords=["color", "depth"]
                )
                _path_datas += [{"color_pth": col_pth, "depth_pth": dep_pth} for col_pth, dep_pth in zip(paths["color"], paths["depth"])]

        if (alternative_path == self._data_path) or (alternative_path is None):
            self._path_datas = _path_datas
        
        return dpcpy(_path_datas)

