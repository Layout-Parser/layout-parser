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

import cv2
import pytest
from layoutparser import PaddleDetectionLayoutModel

def test_only_effdet_model():

    # When all the backeds are not installed, it should 
    # elicit only ImportErrors
    
    config = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    model = PaddleDetectionLayoutModel(config)
    image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
    layout = model.detect(image)
    
    with pytest.raises(ImportError):
        from layoutparser import EfficientDetLayoutModel
        from layoutparser import Detectron2LayoutModel