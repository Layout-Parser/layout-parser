from layoutparser.models import *
import cv2

ALL_CONFIGS = [
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
]


def test_Detectron2Model(is_large_scale=False):

    if is_large_scale:

        for config in ALL_CONFIGS:
            model = Detectron2LayoutModel(config)

            image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
            layout = model.detect(image)
    else:
        model = Detectron2LayoutModel("tests/fixtures/model/config.yml")
        image = cv2.imread("tests/fixtures/model/test_model_image.jpg")
        layout = model.detect(image)