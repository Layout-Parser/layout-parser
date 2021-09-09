__version__ = "0.2.0"

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
        "load_dataframe"
    ],
    "file_utils":[
        "is_torch_available",
        "is_torch_cuda_available",
        "is_detectron2_available",
        "is_paddle_available",
        "is_pytesseract_available",
        "is_gcv_available",
        "requires_backends"
    ]
}

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
