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

from layoutparser.elements import *
from layoutparser.ocr import *
from layoutparser.visualization import *
import cv2
import numpy as np


def test_viz():

    image = cv2.imread("tests/fixtures/ocr/test_gcv_image.jpg")
    ocr_agent = GCVAgent.with_credential(
        "tests/fixtures/ocr/test_gcv_credential.json", languages=["en"]
    )
    res = ocr_agent.load_response("tests/fixtures/ocr/test_gcv_response.json")

    draw_box(image, Layout([]))
    draw_text(image, Layout([]))

    layout = Layout(
        [
            Interval(0, 10, axis="x"),
            Rectangle(0, 50, 100, 80),
            Quadrilateral(np.array([[10, 10], [30, 40], [90, 40], [10, 20]])),
        ]
    )

    draw_box(image, layout)
    draw_text(image, layout)

    # Test colors
    draw_box(image, layout, box_color=["red", "green", "blue"])
    draw_text(image, layout, box_color=["red", "green", "blue"])
    with pytest.raises(ValueError):
        draw_box(image, layout, box_color=["red", "green", "blue", "yellow"])
    with pytest.raises(ValueError):
        draw_text(image, layout, box_color=["red", "green", "blue", "yellow"], with_layout=True)

    for idx, level in enumerate(
        [
            GCVFeatureType.SYMBOL,
            GCVFeatureType.WORD,
            GCVFeatureType.PARA,
            GCVFeatureType.BLOCK,
            GCVFeatureType.PAGE,
        ]
    ):

        layout = ocr_agent.gather_full_text_annotation(res, level)

        draw_text(
            image,
            layout,
            arrangement="ud" if idx % 2 else "ud",
            font_size=15,
            text_color="pink",
            text_background_color="grey",
            text_background_alpha=0.1,
            with_box_on_text=True,
            text_box_width=2,
            text_box_color="yellow",
            text_box_alpha=0.2,
            with_layout=True,
            box_width=1,
            color_map={None: "blue"},
            show_element_id=True,
            id_font_size=8,
            box_alpha=0.25,
            id_text_background_alpha=0.25,
        )

        draw_box(image, layout)
        draw_text(image, layout)
