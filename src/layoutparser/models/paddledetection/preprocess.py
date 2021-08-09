import cv2
import numpy as np


def decode_image(image, im_info):
    """Args:
        image (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
    Returns:
        image (np.ndarray): processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    im_info['im_shape'] = np.array(image.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    return image, im_info

def resize(image, im_info):
    """Args:
        image (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
    Returns:
        image (np.ndarray): processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    target_size = [640, 640]
    im_channel = image.shape[2]
    origin_shape = image.shape[:2]
    im_c = image.shape[2]

    resize_h, resize_w = target_size
    # im_scale_y: the resize ratio of Y
    im_scale_y = resize_h / float(origin_shape[0])
    # the resize ratio of X
    im_scale_x = resize_w / float(origin_shape[1])

    # set image_shape
    im_info['input_shape'][1] = int(im_scale_y * image.shape[0])
    im_info['input_shape'][2] = int(im_scale_x * image.shape[1])
    image = cv2.resize(
        image,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=2)
    im_info['im_shape'] = np.array(image.shape[:2]).astype('float32')
    im_info['scale_factor'] = np.array(
        [im_scale_y, im_scale_x]).astype('float32')
    return image, im_info

def normalize_image(image,
        im_info,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_scale=True):
    """Args:
        image (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
        mean (list): image - mean
        std (list): image / std
        is_scale (bool): whether need image / 255
    Returns:
        image (np.ndarray): processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    image = image.astype(np.float32, copy=False)
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]

    if is_scale:
        image = image / 255.0

    image -= mean
    image /= std
    return image, im_info

def permute(image, im_info):
    """Args:
        image (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
    Returns:
        image (np.ndarray): processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    image = image.transpose((2, 0, 1)).copy()
    return image, im_info
