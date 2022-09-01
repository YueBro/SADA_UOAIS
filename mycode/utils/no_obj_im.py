import cv2


__all__ = [
    "get_no_obj_im"
]


def get_no_obj_im(width_height=(640, 480)):
    im = cv2.imread("mycode/utils/_no_obj.png")
    im = cv2.resize(im, width_height)
    return im
