from ..tools import get_img_paths


class UOAIS_Dataloader:
    __doc__ = \
    """
    
    """

    def __init__(self, data_path='datasets/UOAIS-Sim'):
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
                    'vis_pths': [str, ...]
                }, ...
            ]
        """
        if (alternative_path == self._data_path) or (alternative_path is None):
            path = self._data_path
            self._path_datas = _load_uoais_paths_as_dict(path=path)
            return self._path_datas
        else:
            path = alternative_path
            return _load_uoais_paths_as_dict(path=path)


def _load_uoais_paths_as_dict(path='datasets/UOAIS-Sim'):

    paths = get_img_paths(
        path + "/val/bin/color",
        path + "/val/bin/depth",
        path + "/val/tabletop/color",
        path + "/val/tabletop/depth",
        keywords=["color1", "depth1", "color2", "depth2"]
    )

    assert len(paths)>0, "Data path incorrect."

    paths = {
        "color": paths["color1"] + paths["color2"],
        "depth": paths["depth1"] + paths["depth2"],
    }

    paths = [{"color_pth": col, "depth_pth": dep} for col, dep in zip(paths["color"], paths["depth"])]

    return paths
