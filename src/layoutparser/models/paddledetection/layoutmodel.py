# Copyright 2021 The Layout Parser team and Paddle Detection model 
# contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union, Dict, Any, Tuple
import os
from functools import reduce
import warnings

from PIL import Image
import cv2
import numpy as np

from .catalog import PathManager, LABEL_MAP_CATALOG, MODEL_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

from ...file_utils import is_paddle_available

if is_paddle_available():
    import paddle.inference


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
        device(:obj:`str`, optional):
            Whether to use cuda or cpu devices. If not set, LayoutParser will
            automatically determine the device to initialize the models on.
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
                                    lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config')
        >>> model.detect(image)

    """

    DEPENDENCIES = ["paddle"]
    DETECTOR_NAME = "paddledetection"
    MODEL_CATALOG = MODEL_CATALOG

    def __init__(
        self,
        config_path=None,
        model_path=None,
        label_map=None,
        device=None,
        enforce_cpu=None,
        extra_config=None,
    ):

        if enforce_cpu is not None:
            warnings.warn(
                "Setting enforce_cpu is deprecated. Please set `device` instead.",
                DeprecationWarning,
            )

        if extra_config is None:
            extra_config = {}

        _, model_path = self.config_parser(config_path, model_path)
        model_dir = PathManager.get_local_path(model_path)

        if label_map is None:
            if model_path.startswith("lp://"):
                dataset_name = model_path.lstrip("lp://").split("/")[1]
                label_map = LABEL_MAP_CATALOG[dataset_name]
            else:
                label_map = {}

        self.label_map = label_map

        # TODO: rethink how to save store the default constants
        self.predictor = self.load_predictor(
            model_dir,
            device=device,
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

    def load_predictor(
        self,
        model_dir,
        device=None,
        enable_mkldnn=False,
        thread_num=10,
    ):
        """set AnalysisConfig, generate AnalysisPredictor
        Args:
            model_dir (str): root path of __model__ and __params__
            device (str): cuda or cpu
        Returns:
            predictor (PaddlePredictor): AnalysisPredictor
        Raises:
            ValueError: predict by TensorRT need enforce_cpu == False.
        """

        config = paddle.inference.Config(
            os.path.join(
                model_dir, "inference.pdmodel"
            ),  # TODO: Move them to some constants
            os.path.join(model_dir, "inference.pdiparams"),
        )

        if device == "cuda":
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
        predictor = paddle.inference.create_predictor(config)
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

            cur_block = TextBlock(
                Rectangle(x_1, y_1, x_2, y_2),
                type=self.label_map.get(clsid, clsid),
                score=score,
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
        image = self.image_loader(image)

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

    def image_loader(self, image: Union["np.ndarray", "Image.Image"]):

        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image)

        return image
