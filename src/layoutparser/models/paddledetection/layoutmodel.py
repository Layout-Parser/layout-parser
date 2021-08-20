from typing import List, Union, Dict, Any, Tuple
import os
from functools import reduce

from PIL import Image
import cv2
import numpy as np

from .catalog import PathManager, LABEL_MAP_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout


__all__ = ["PaddleDetectionLayoutModel"]


def _resize_image(
    image: np.ndarray, target_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image (np.ndarray): image (np.ndarray)
    Returns:
        image (np.ndarray): processed image (np.ndarray)
    """

    origin_shape = image.shape[:2]

    resize_h, resize_w = target_size
    # im_scale_y: the resize ratio of Y
    im_scale_y = resize_h / float(origin_shape[0])
    # the resize ratio of X
    im_scale_x = resize_w / float(origin_shape[1])

    # resize image
    image = cv2.resize(image, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=2)
    scale_factor = np.array([im_scale_y, im_scale_x]).astype("float32")
    return image, scale_factor


class PaddleDetectionLayoutModel(BaseLayoutModel):
    """Create a PaddleDetection-based Layout Detection Model

    Args:
        config_path (:obj:`str`):
            The path to the configuration file.
        model_path (:obj:`str`, None):
            The path to the saved weights of the model.
            If set, overwrite the weights in the configuration file.
            Defaults to `None`.
        label_map (:obj:`dict`, optional):
            The map from the model prediction (ids) to real
            word labels (strings). If the config is from one of the supported
            datasets, Layout Parser will automatically initialize the label_map.
            Defaults to `None`.
        enforce_cpu(:obj:`bool`, optional):
            When set to `True`, it will enforce using cpu even if it is on a CUDA
            available device.
        extra_config (:obj:`dict`, optional):
            Extra configuration passed to the PaddleDetection model configuration.
            Defaults to `{}`.
            Including arguments:
            enable_mkldnn (:obj:`bool`, optional):
                Whether use mkldnn to accelerate the computation.
                Defaults to False.
            thread_num (:obj:`int`, optional):
                The number of threads.
                Defaults to 10.
            threshold (:obj:`float`, optional):
                Threshold to reserve the result for output.
                Defaults to 0.5.
            target_size (:obj:`list`, optional):
                The image shape after resize.
                Defaults to [640,640].

    Examples::
        >>> import layoutparser as lp
        >>> model = lp.models.PaddleDetectionLayoutModel('
                                    lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config')
        >>> model.detect(image)

    """

    DEPENDENCIES = ["paddlepaddle"]
    MODULES = [
        {
            "import_name": "_inference",
            "module_path": "paddle.inference",
        },
    ]
    DETECTOR_NAME = "paddledetection"

    def __init__(
        self,
        config_path=None,
        model_path=None,
        label_map=None,
        enforce_cpu=False,
        extra_config=None,
    ):
    
        if extra_config is None:
            extra_config = {}

        if model_path is not None:
            model_dir = model_path
        elif config_path is not None and config_path.startswith(
            "lp://"
        ):  # TODO: Move "lp://" to a constant
            if label_map is None:
                dataset_name = config_path.lstrip("lp://").split("/")[0]
                label_map = LABEL_MAP_CATALOG[dataset_name]
            config_path = self._reconstruct_path_with_detector_name(config_path)
            model_dir = PathManager.get_local_path(config_path)
        else:
            raise Exception("Please set config_path or model_path first")

        # TODO: rethink how to save store the default constants
        self.predictor = self.load_predictor(
            model_dir,
            enforce_cpu=enforce_cpu,
            enable_mkldnn=extra_config.get("enable_mkldnn", False),
            thread_num=extra_config.get("thread_num", 10),
        )

        self.threshold = extra_config.get("threshold", 0.5)
        self.target_size = extra_config.get("target_size", [640, 640])
        self.pixel_mean = extra_config.get(
            "pixel_mean", np.array([[[0.485, 0.456, 0.406]]])
        )
        self.pixel_std = extra_config.get(
            "pixel_std", np.array([[[0.229, 0.224, 0.225]]])
        )
        self.label_map = label_map

    def _reconstruct_path_with_detector_name(self, path: str) -> str:
        """This function will add the detector name (paddledetection) into the
        lp model config path to get the "canonical" model name.

        For example,
        for a given config_path `lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config`,it will
        transform it into `lp://paddledetection/PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config`.
        However, if the config_path already contains the detector name, we won't change it.

        This function is a general step to support multiple backends in the layout-parser
        library.

        Args:
            path (str): The given input path that might or might not contain the detector name.

        Returns:
            str: a modified path that contains the detector name.
        """
        if path.startswith("lp://"):  # TODO: Move "lp://" to a constant
            model_name = path[len("lp://") :]
            model_name_segments = model_name.split("/")
            if (
                len(model_name_segments) == 3
                and "paddledetection" not in model_name_segments
            ):
                return "lp://" + self.DETECTOR_NAME + "/" + path[len("lp://") :]
        return path

    def load_predictor(
        self,
        model_dir,
        enforce_cpu=False,
        enable_mkldnn=False,
        thread_num=10,
    ):
        """set AnalysisConfig, generate AnalysisPredictor
        Args:
            model_dir (str): root path of __model__ and __params__
            enforce_cpu (bool): whether use cpu
        Returns:
            predictor (PaddlePredictor): AnalysisPredictor
        Raises:
            ValueError: predict by TensorRT need enforce_cpu == False.
        """

        config = self._inference.Config(
            os.path.join(
                model_dir, "inference.pdmodel"
            ),  # TODO: Move them to some constants
            os.path.join(model_dir, "inference.pdiparams"),
        )

        if not enforce_cpu:
            # initial GPU memory(M), device ID
            # 2000 is an appropriate value for PaddleDetection model
            config.enable_use_gpu(2000, 0)
            # optimize graph and fuse op
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(thread_num)
            if enable_mkldnn:
                config.enable_mkldnn()
                try:
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    config.set_mkldnn_cache_capacity(10)
                    config.enable_mkldnn()
                except Exception as e:
                    print(
                        "The current environment does not support `mkldnn`, so disable mkldnn."
                    )

        # disable print log when predict
        config.disable_glog_info()
        # enable shared memory
        config.enable_memory_optim()
        # disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
        predictor = self._inference.create_predictor(config)
        return predictor

    def preprocess(self, image):
        """preprocess image

        Args:
            image (np.ndarray): image (np.ndarray)
        Returns:
            inputs (dict): input of model
        """

        # resize image by target_size and max_size
        image, scale_factor = _resize_image(image, self.target_size)
        input_shape = np.array(image.shape[:2]).astype("float32")
        # normalize image
        image = (image / 255.0 - self.pixel_mean) / self.pixel_std
        # transpose images
        image = image.transpose((2, 0, 1)).copy()

        inputs = {}
        inputs["image"] = np.array(image)[np.newaxis, :].astype("float32")
        inputs["im_shape"] = np.array(input_shape)[np.newaxis, :].astype("float32")
        inputs["scale_factor"] = np.array(scale_factor)[np.newaxis, :].astype("float32")
        return inputs

    def gather_output(self, np_boxes):
        """process output"""
        layout = Layout()
        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print("[WARNING] No object detected.")
            results = {"boxes": np.array([])}
        else:
            results = {}
            results["boxes"] = np_boxes

        np_boxes = results["boxes"]
        expect_boxes = (np_boxes[:, 1] > self.threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]

        for np_box in np_boxes:
            clsid, bbox, score = int(np_box[0]), np_box[2:], np_box[1]
            x_1, y_1, x_2, y_2 = bbox

            if self.label_map is not None:
                label = self.label_map[clsid]

            cur_block = TextBlock(
                Rectangle(x_1, y_1, x_2, y_2), type=label, score=score
            )
            layout.append(cur_block)

        return layout

    def detect(self, image):
        """Detect the layout of a given image.

        Args:
            image (:obj:`np.ndarray` or `PIL.Image`): The input image to detect.

        Returns:
            :obj:`~layoutparser.Layout`: The detected layout of the input image
        """

        # Convert PIL Image Input
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image)

        inputs = self.preprocess(image)
        
        input_names = self.predictor.get_input_names()

        for input_name in input_names:
            input_tensor = self.predictor.get_input_handle(input_name)
            input_tensor.copy_from_cpu(inputs[input_name])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()

        layout = self.gather_output(np_boxes)
        return layout