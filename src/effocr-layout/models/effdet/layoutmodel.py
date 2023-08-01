# Copyright 2021 The Layout Parser team. All rights reserved.
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

from typing import List, Optional, Union, Dict, Any, Tuple

from PIL import Image
import numpy as np

from .catalog import PathManager, LABEL_MAP_CATALOG, MODEL_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

from ...file_utils import is_effdet_available, is_torch_cuda_available

if is_effdet_available():
    import torch
    from effdet import create_model
    from effdet.data.transforms import (
        IMAGENET_DEFAULT_MEAN,
        IMAGENET_DEFAULT_STD,
        transforms_coco_eval,
    )
else:
    # Copied from https://github.com/rwightman/efficientdet-pytorch/blob/c5b694aa34900fdee6653210d856ca8320bf7d4e/effdet/data/transforms.py#L13 
    # Such that when effdet is not loaded, we'll still have default values for IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    # IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
    # IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


class InputTransform:
    def __init__(
        self,
        image_size,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):

        self.mean = mean
        self.std = std

        self.transform = transforms_coco_eval(
            image_size,
            interpolation="bilinear",
            use_prefetcher=True,
            fill_color="mean",
            mean=self.mean,
            std=self.std,
        )

        self.mean_tensor = torch.tensor([x * 255 for x in mean]).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor([x * 255 for x in std]).view(1, 3, 1, 1)

    def preprocess(self, image: Image) -> Tuple["torch.Tensor", Dict]:

        image = image.convert("RGB")
        image_info = {"img_size": image.size}

        input, image_info = self.transform(image, image_info)
        image_info = {
            key: torch.tensor(val).unsqueeze(0) for key, val in image_info.items()
        }

        input = torch.tensor(input).unsqueeze(0)
        input = input.float().sub_(self.mean_tensor).div_(self.std_tensor)

        return input, image_info


class EfficientDetLayoutModel(BaseLayoutModel):
    """Create a EfficientDet-based Layout Detection Model

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
            Extra configuration passed to the EfficientDet model
            configuration. Currently supported arguments:
                num_classes: specifying the number of classes for the models
                output_confidence_threshold: minmum object prediction confidence to retain

    Examples::
        >>> import layoutparser as lp
        >>> model = lp.EfficientDetLayoutModel("lp://PubLayNet/tf_efficientdet_d0/config")
        >>> model.detect(image)

    """

    DEPENDENCIES = ["effdet"]
    DETECTOR_NAME = "efficientdet"
    MODEL_CATALOG = MODEL_CATALOG

    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.25

    def __init__(
        self,
        config_path: str,
        model_path: str = None,
        label_map: Optional[Dict] = None,
        extra_config: Optional[Dict] = None,
        enforce_cpu: bool = False,
        device: str = None,
    ):

        if is_torch_cuda_available():
            if device is None:
                device = "cuda"
        else:
            device = "cpu"
        self.device = device

        extra_config = extra_config if extra_config is not None else {}

        self._initialize_model(config_path, model_path, label_map, extra_config)

        self.output_confidence_threshold = extra_config.get(
            "output_confidence_threshold", self.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
        )

        self.preprocessor = InputTransform(self.config.image_size)

    def _initialize_model(
        self,
        config_path: str,
        model_path: Optional[str],
        label_map: Optional[Dict],
        extra_config: Optional[Dict],
    ):

        config_path, model_path = self.config_parser(config_path, model_path)

        if config_path.startswith("lp://"):
            # If it's officially supported by layoutparser
            dataset_name, model_name = config_path.lstrip("lp://").split("/")[1:3]

            if label_map is None:
                label_map = LABEL_MAP_CATALOG[dataset_name]
            num_classes = len(label_map)

            model_path = PathManager.get_local_path(model_path)

            self.model = create_model(
                model_name,
                num_classes=num_classes,
                bench_task="predict",
                pretrained=True,
                checkpoint_path=model_path,
            )
        else:
            assert (
                model_path is not None
            ), f"When the specified model is not layoutparser-based, you need to specify the model_path"

            assert (
                label_map is not None or "num_classes" in extra_config
            ), "When the specified model is not layoutparser-based, you need to specify the label_map or add num_classes in the extra_config"

            model_name = config_path
            model_path = PathManager.get_local_path(
                model_path
            )  # It might be an https URL

            num_classes = len(label_map) if label_map else extra_config["num_classes"]

            self.model = create_model(
                model_name,
                num_classes=num_classes,
                bench_task="predict",
                pretrained=True,
                checkpoint_path=model_path,
            )

        self.model.to(self.device)
        self.model.eval()
        self.config = self.model.config
        self.label_map = label_map if label_map is not None else {}

    def detect(self, image: Union["np.ndarray", "Image.Image"]):

        image = self.image_loader(image)

        model_inputs, image_info = self.preprocessor.preprocess(image)

        model_outputs = self.model(
            model_inputs.to(self.device),
            {key: val.to(self.device) for key, val in image_info.items()},
        )

        layout = self.gather_output(model_outputs)
        return layout

    def gather_output(self, model_outputs: "torch.Tensor") -> Layout:

        model_outputs = model_outputs.cpu().detach()
        box_predictions = Layout()

        for index, sample in enumerate(model_outputs):
            sample[:, 2] -= sample[:, 0]
            sample[:, 3] -= sample[:, 1]

            for det in sample:

                score = float(det[4])
                pred_cat = int(det[5])
                x, y, w, h = det[0:4].tolist()

                if (
                    score < self.output_confidence_threshold
                ):  # stop when below this threshold, scores in descending order
                    break

                box_predictions.append(
                    TextBlock(
                        block=Rectangle(x, y, w + x, h + y),
                        score=score,
                        id=index,
                        type=self.label_map.get(pred_cat, pred_cat),
                    )
                )

        return box_predictions

    def image_loader(self, image: Union["np.ndarray", "Image.Image"]):
        
        # Convert cv2 Image Input
        if isinstance(image, np.ndarray):
            # In this case, we assume the image is loaded by cv2
            # and the channel order is BGR
            image = image[..., ::-1]
            image = Image.fromarray(image, mode="RGB")

        return image
