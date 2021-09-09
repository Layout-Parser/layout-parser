from layoutparser import load_json
from layoutparser.models import *
import cv2

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
    "lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
    "lp://TableBank/ppyolov2_r50vd_dcn_365e_tableBank_word/config",
    "lp://TableBank/ppyolov2_r50vd_dcn_365e_tableBank_latex/config",
]

ALL_EFFDET_MODEL_CONFIGS = [
    "lp://PubLayNet/tf_efficientdet_d0/config",
    "lp://PubLayNet/tf_efficientdet_d1/config",
    "lp://MFD/tf_efficientdet_d0/config",
    "lp://MFD/tf_efficientdet_d1/config",
]

def test_Detectron2Model(is_large_scale=False):

    if is_large_scale:

        for config in ALL_DETECTRON2_MODEL_CONFIGS:
            model = Detectron2LayoutModel(config)

            image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
            layout = model.detect(image)
    else:
        model = Detectron2LayoutModel("tests/fixtures/model/config.yml")
        image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
        layout = model.detect(image)
        
    # Test in enforce CPU mode
    model = Detectron2LayoutModel("tests/fixtures/model/config.yml", enforce_cpu=True)
    image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
    layout = model.detect(image)
    
def test_Detectron2Model_version_compatibility(enabled=False):
    
    if enabled:
        model = Detectron2LayoutModel(
            config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.85,
                "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.75,
            ],
        )
        image = cv2.imread("tests/fixtures/model/layout_detection_reference.jpg")
        layout = model.detect(image)
        assert load_json("tests/fixtures/model/layout_detection_reference.json") == layout

def test_PaddleDetectionModel(is_large_scale=False):
    """ test PaddleDetection model """
    if is_large_scale:

        for config in ALL_PADDLEDETECTION_MODEL_CONFIGS:
            model = PaddleDetectionLayoutModel(config)

            image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
            layout = model.detect(image)
    else:
        model = PaddleDetectionLayoutModel("lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config")
        image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
        layout = model.detect(image)
        
    # Test in enforce CPU mode
    model = PaddleDetectionLayoutModel("lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config", enforce_cpu=True)
    image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
    layout = model.detect(image)

def test_EffDetModel(is_large_scale=False):

    if is_large_scale:

        for config in ALL_EFFDET_MODEL_CONFIGS:
            model = EfficientDetLayoutModel(config)

            image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
            layout = model.detect(image)
    else:
        model = EfficientDetLayoutModel("lp://PubLayNet/tf_efficientdet_d0/config")
        image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
        layout = model.detect(image)