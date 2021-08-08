import cv2
import numpy as np


def decode_image(im_file, im_info):
    """read rgb image
    Args:
        im_file (str|np.ndarray): input can be image path or np.ndarray
        im_info (dict): info of image
    Returns:
        image (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as file:
            im_read = file.read()
        data = np.frombuffer(im_read, dtype='uint8')
        image = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = im_file
    im_info['im_shape'] = np.array(image.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    return image, im_info


class Resize(object):
    """resize image by target_size and max_size
    Args:
        target_size (int): the target size of image
        keep_ratio (bool): whether keep_ratio or not, default true
        interp (int): method of resize
    """

    def __init__(
            self,
            target_size=[640, 640],
            keep_ratio=False,
            interp=2, ):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, image, im_info):
        """
        Args:
            image (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            image (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        im_channel = image.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(image)
        # set image_shape
        im_info['input_shape'][1] = int(im_scale_y * image.shape[0])
        im_info['input_shape'][2] = int(im_scale_x * image.shape[1])
        image = cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        im_info['im_shape'] = np.array(image.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array(
            [im_scale_y, im_scale_x]).astype('float32')
        return image, im_info

    def generate_scale(self, image):
        """
        Args:
            image (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = image.shape[:2]
        im_c = image.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x


class NormalizeImage(object):
    """normalize image
    Args:
        mean (list): image - mean
        std (list): image / std
        is_scale (bool): whether need image / 255
        is_channel_first (bool): if True: image shape is CHW, else: HWC
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_scale=True):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale

    def __call__(self, image, im_info):
        """
        Args:
            image (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            image (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        image = image.astype(np.float32, copy=False)
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]

        if self.is_scale:
            image = image / 255.0

        image -= mean
        image /= std
        return image, im_info


class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(self, ):
        super(Permute, self).__init__()

    def __call__(self, image, im_info):
        """
        Args:
            image (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            image (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        image = image.transpose((2, 0, 1)).copy()
        return image, im_info


class PadStride(object):
    """ padding image for model with FPN ,
        instead PadBatch(pad_to_stride, pad_gt) in original config
    Args:
        stride (bool): model with FPN need image shape % stride == 0
    """

    def __init__(self, stride=0):
        self.coarsest_stride = stride

    def __call__(self, image, im_info):
        """
        Args:
            image (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            image (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        coarsest_stride = self.coarsest_stride
        if coarsest_stride <= 0:
            return image, im_info
        im_c, im_h, im_w = image.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = image
        return padding_im, im_info


def preprocess(image, input_shape):
    """ process image by preprocess_ops """
    im_info = {
        'scale_factor': np.array(
            [1., 1.], dtype=np.float32),
        'im_shape': None,
        'input_shape': input_shape,
    }
    image, im_info = decode_image(image, im_info)
    resize = Resize()
    permute = Permute()
    normalize_image = NormalizeImage()
    image, im_info = resize(image, im_info)
    image, im_info = normalize_image(image, im_info)
    image, im_info = permute(image, im_info)
    return image, im_info
    