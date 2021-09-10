from .detectron2.layoutmodel import Detectron2LayoutModel
from .paddledetection.layoutmodel import PaddleDetectionLayoutModel
from .effdet.layoutmodel import EfficientDetLayoutModel
from .model_config import (
    is_lp_layout_model_config_any_format,
    add_identifier_for_config,
    layout_model_config_parser,
)

ALL_AVAILABLE_BACKENDS = {
    Detectron2LayoutModel.DETECTOR_NAME: Detectron2LayoutModel,
    PaddleDetectionLayoutModel.DETECTOR_NAME: PaddleDetectionLayoutModel,
    EfficientDetLayoutModel.DETECTOR_NAME: EfficientDetLayoutModel,
}


def AutoLayoutModel(
    config_path, model_path=None, label_map=None, extra_config=None, device=None
):
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