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

from iopath.common.file_io import PathHandler

from ..base_catalog import PathManager

MODEL_CATALOG = {
    "HJDataset": {
        "faster_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/HJDataset/faster_rcnn_R_50_FPN_3x/model_final.pth",
        "mask_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/HJDataset/mask_rcnn_R_50_FPN_3x/model_final.pth",
        "retinanet_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/HJDataset/retinanet_R_50_FPN_3x/model_final.pth",
    },
    "PubLayNet": {
        "faster_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/PubLayNet/faster_rcnn_R_50_FPN_3x/model_final.pth",
        "mask_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/PubLayNet/mask_rcnn_R_50_FPN_3x/model_final.pth",
        "mask_rcnn_X_101_32x8d_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/model_final.pth",
    },
    "PrimaLayout": {
        "mask_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/PrimaLayout/mask_rcnn_R_50_FPN_3x/model_final.pth"
    },
    "NewspaperNavigator": {
        "faster_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/NewspaperNavigator/faster_rcnn_R_50_FPN_3x/model_final.pth",
    },
    "TableBank": {
        "faster_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/TableBank/faster_rcnn_R_50_FPN_3x/model_final.pth",
        "faster_rcnn_R_101_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/TableBank/faster_rcnn_R_101_FPN_3x/model_final.pth",
    },
    "MFD": {
        "faster_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/MFD/faster_rcnn_R_50_FPN_3x/model_final.pth",
    },
}

CONFIG_CATALOG = {
    "HJDataset": {
        "faster_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/HJDataset/faster_rcnn_R_50_FPN_3x/config.yml",
        "mask_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/HJDataset/mask_rcnn_R_50_FPN_3x/config.yml",
        "retinanet_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/HJDataset/retinanet_R_50_FPN_3x/config.yml",
    },
    "PubLayNet": {
        "faster_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/PubLayNet/faster_rcnn_R_50_FPN_3x/config.yml",
        "mask_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/PubLayNet/mask_rcnn_R_50_FPN_3x/config.yml",
        "mask_rcnn_X_101_32x8d_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config.yml",
    },
    "PrimaLayout": {
        "mask_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/PrimaLayout/mask_rcnn_R_50_FPN_3x/config.yml"
    },
    "NewspaperNavigator": {
        "faster_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config.yml",
    },
    "TableBank": {
        "faster_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/TableBank/faster_rcnn_R_50_FPN_3x/config.yml",
        "faster_rcnn_R_101_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/TableBank/faster_rcnn_R_101_FPN_3x/config.yml",
    },
    "MFD": {
        "faster_rcnn_R_50_FPN_3x": "https://huggingface.co/layoutparser/detectron2/resolve/main/MFD/faster_rcnn_R_50_FPN_3x/config.yml",
    },
}

# fmt: off
LABEL_MAP_CATALOG = {
    "HJDataset": {
        1: "Page Frame",
        2: "Row",
        3: "Title Region",
        4: "Text Region",
        5: "Title",
        6: "Subtitle",
        7: "Other",
    },
    "PubLayNet": {
        0: "Text", 
        1: "Title", 
        2: "List", 
        3: "Table", 
        4: "Figure"},
    "PrimaLayout": {
        1: "TextRegion",
        2: "ImageRegion",
        3: "TableRegion",
        4: "MathsRegion",
        5: "SeparatorRegion",
        6: "OtherRegion",
    },
    "NewspaperNavigator": {
        0: "Photograph",
        1: "Illustration",
        2: "Map",
        3: "Comics/Cartoon",
        4: "Editorial Cartoon",
        5: "Headline",
        6: "Advertisement",
    },
    "TableBank": {
        0: "Table"
    },
    "MFD": {
        1: "Equation"
    },
}
# fmt: on


class LayoutParserDetectron2ModelHandler(PathHandler):
    """
    Resolve anything that's in LayoutParser model zoo.
    """

    PREFIX = "lp://detectron2/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        model_name = path[len(self.PREFIX) :]

        dataset_name, *model_name, data_type = model_name.split("/")

        if data_type == "weight":
            model_url = MODEL_CATALOG[dataset_name]["/".join(model_name)]
        elif data_type == "config":
            model_url = CONFIG_CATALOG[dataset_name]["/".join(model_name)]
        else:
            raise ValueError(f"Unknown data_type {data_type}")
        return PathManager.get_local_path(model_url, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(LayoutParserDetectron2ModelHandler())
