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

from typing import Optional, Dict, Union, List
from .detectron2.layoutmodel import Detectron2LayoutModel
from .paddledetection.layoutmodel import PaddleDetectionLayoutModel
from .effdet.layoutmodel import EfficientDetLayoutModel
from .model_config import (
    is_lp_layout_model_config_any_format,
)

ALL_AVAILABLE_BACKENDS = {
    Detectron2LayoutModel.DETECTOR_NAME: Detectron2LayoutModel,
    PaddleDetectionLayoutModel.DETECTOR_NAME: PaddleDetectionLayoutModel,
    EfficientDetLayoutModel.DETECTOR_NAME: EfficientDetLayoutModel,
}


def AutoLayoutModel(
    config_path: str,
    model_path: Optional[str] = None,
    label_map: Optional[Dict]=None,
    device: Optional[str]=None,
    extra_config: Optional[Union[Dict, List]]=None,
) -> "BaseLayoutModel":
    """[summary]

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
            Extra configuration passed used for initializing the layout model.

    Returns:
        # BaseLayoutModel: the create LayoutModel instance
    """
    if not is_lp_layout_model_config_any_format(config_path):
        raise ValueError(f"Invalid model config_path {config_path}")
    for backend_name in ALL_AVAILABLE_BACKENDS:
        if backend_name in config_path:
            return ALL_AVAILABLE_BACKENDS[backend_name](
                config_path,
                model_path=model_path,
                label_map=label_map,
                extra_config=extra_config,
                device=device,
            )
