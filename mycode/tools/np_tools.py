import numpy as np


def img_interp(img):
    return np.interp(img, (np.min(img), np.max(img)), (0,255))
