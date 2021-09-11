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

import pytest
import cv2

from layoutparser import load_json
from layoutparser.models import *

ALL_DETECTRON2_MODEL_CONFIGS = [
    "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config",
    "lp://HJDataset/faster_rcnn_R_50_FPN_3x/config",
    "lp://HJDataset/mask_rcnn_R_50_FPN_3x/config",
    "lp://HJDataset/retinanet_R_50_FPN_3x/config",
    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    "lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config",
    "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
    "lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config",
    "lp://TableBank/faster_rcnn_R_50_FPN_3x/config",
    "lp://TableBank/faster_rcnn_R_101_FPN_3x/config",
    "lp://MFD/faster_rcnn_R_50_FPN_3x/config",
]

ALL_PADDLEDETECTION_MODEL_CONFIGS = [
    "lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
    "lp://TableBank/ppyolov2_r50vd_dcn_365e/config",
]

ALL_EFFDET_MODEL_CONFIGS = [
    "lp://PubLayNet/tf_efficientdet_d0/config",
    "lp://PubLayNet/tf_efficientdet_d1/config",
    "lp://MFD/tf_efficientdet_d0/config",
    "lp://MFD/tf_efficientdet_d1/config",
]

AUTOMODEL_CONFIGS = [
    "lp://detectron2/PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    "lp://paddledetection/PubLayNet/ppyolov2_r50vd_dcn_365e/config",
    "lp://efficientdet/PubLayNet/tf_efficientdet_d0/config",
]

def _construct_valid_config_variations(config, backend_name):
    dataset_name, arch_name, identifier = config[len("lp://") :].split("/")
    return [
        "lp://" + "/".join([backend_name, dataset_name, arch_name, identifier]),
        "lp://" + "/".join([backend_name, dataset_name, arch_name]),
        "lp://" + "/".join([backend_name, dataset_name]),
        "lp://" + "/".join([dataset_name, arch_name, identifier]),
        "lp://" + "/".join([dataset_name, arch_name]),
        "lp://" + "/".join([dataset_name]),
    ]


def _construct_invalid_config_variations(config, backend_name):
    dataset_name, arch_name, identifier = config[len("lp://") :].split("/")
    return [
        "lp://" + "/".join([backend_name]),
    ]


def _single_config_test_pipeline(TestLayoutModel, base_config):
    for config in _construct_valid_config_variations(
        base_config, TestLayoutModel.DETECTOR_NAME
    ):
        model = TestLayoutModel(config)
        image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
        layout = model.detect(image)
        del model

    for config in _construct_invalid_config_variations(
        base_config, TestLayoutModel.DETECTOR_NAME
    ):
        with pytest.raises(ValueError):
            model = TestLayoutModel(config)


def test_Detectron2Model(is_large_scale=False):

    if is_large_scale:

        for config in ALL_DETECTRON2_MODEL_CONFIGS:
            model = Detectron2LayoutModel(config)

            image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
            layout = model.detect(image)
    else:
        _single_config_test_pipeline(
            Detectron2LayoutModel, ALL_DETECTRON2_MODEL_CONFIGS[0]
        )
    # Test in enforce CPU mode
    model = Detectron2LayoutModel("tests/fixtures/model/config.yml")
    image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
    layout = model.detect(image)


def test_Detectron2Model_version_compatibility(enabled=False):

    if enabled:
        model = Detectron2LayoutModel(
            config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                0.85,
                "MODEL.ROI_HEADS.NMS_THRESH_TEST",
                0.75,
            ],
        )
        image = cv2.imread("tests/fixtures/model/layout_detection_reference.jpg")
        layout = model.detect(image)
        assert (
            load_json("tests/fixtures/model/layout_detection_reference.json") == layout
        )


def test_PaddleDetectionModel(is_large_scale=False):
    """test PaddleDetection model"""
    if is_large_scale:

        for config in ALL_PADDLEDETECTION_MODEL_CONFIGS:
            model = PaddleDetectionLayoutModel(config)

            image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
            layout = model.detect(image)
    else:
        _single_config_test_pipeline(
            PaddleDetectionLayoutModel, ALL_PADDLEDETECTION_MODEL_CONFIGS[0]
        )


def test_EffDetModel(is_large_scale=False):

    if is_large_scale:

        for config in ALL_EFFDET_MODEL_CONFIGS:
            model = EfficientDetLayoutModel(config)

            image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
            layout = model.detect(image)
    else:
        _single_config_test_pipeline(
            EfficientDetLayoutModel, ALL_EFFDET_MODEL_CONFIGS[0]
        )

def test_AutoModel():
    for config in AUTOMODEL_CONFIGS:
        model = AutoLayoutModel(config)
        image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
        layout = model.detect(image)