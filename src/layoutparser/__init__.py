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

__version__ = "0.3.4"

import sys

from .file_utils import (
    _LazyModule,
    is_detectron2_available,
    is_paddle_available,
    is_effdet_available,
    is_pytesseract_available,
    is_gcv_available,
)

_import_structure = {
    "elements": [
        "Interval", 
        "Rectangle", 
        "Quadrilateral", 
        "TextBlock", 
        "Layout"
    ],
    "visualization": [
        "draw_box", 
        "draw_text"
    ],
    "io": [
        "load_json", 
        "load_dict", 
        "load_csv", 
        "load_dataframe",
        "load_pdf"
    ],
    "file_utils":[
        "is_torch_available",
        "is_torch_cuda_available",
        "is_detectron2_available",
        "is_paddle_available",
        "is_pytesseract_available",
        "is_gcv_available",
        "requires_backends"
    ],
    "tools": [
        "generalized_connected_component_analysis_1d",
        "simple_line_detection",
        "group_textblocks_based_on_category"
    ]
}

_import_structure["models"] = ["AutoLayoutModel"]

if is_detectron2_available():
    _import_structure["models.detectron2"] = ["Detectron2LayoutModel"]

if is_paddle_available():
    _import_structure["models.paddledetection"] = ["PaddleDetectionLayoutModel"]

if is_effdet_available():
    _import_structure["models.effdet"] = ["EfficientDetLayoutModel"]

if is_pytesseract_available():
    _import_structure["ocr.tesseract_agent"] = [
        "TesseractAgent",
        "TesseractFeatureType",
    ]

if is_gcv_available():
    _import_structure["ocr.gcv_agent"] = ["GCVAgent", "GCVFeatureType"]

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
    extra_objects={"__version__": __version__},
)
