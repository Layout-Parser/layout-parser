import cv2
import numpy as np

def resize(image, target_size):
    """
    Args:
        image (np.ndarray): image (np.ndarray)
    Returns:
        image (np.ndarray): processed image (np.ndarray)
    """

    im_channel = image.shape[2]
    origin_shape = image.shape[:2]
    im_c = image.shape[2]

    resize_h, resize_w = target_size
    # im_scale_y: the resize ratio of Y
    im_scale_y = resize_h / float(origin_shape[0])
    # the resize ratio of X
    im_scale_x = resize_w / float(origin_shape[1])

    # resize image
    image = cv2.resize(
        image,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=2)
    scale_factor = np.array(
        [im_scale_y, im_scale_x]).astype('float32')
    return image, scale_factor
