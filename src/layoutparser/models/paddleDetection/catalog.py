from iopath.common.file_io import PathHandler

from ..base_catalog import PathManager


CONFIG_CATALOG = {
    "PubLayNet": {
        "ppyolov2_r50vd_dcn_365e_publaynet": "https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar",
    },
    "TableBank": {
        "ppyolov2_r50vd_dcn_365e_tableBank_word": "https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_word.tar",
        "ppyolov2_r50vd_dcn_365e_tableBank_latex": "https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_latex.tar",
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
}
# fmt: on


class LayoutParserDetectron2ModelHandler(PathHandler):
    """
    Resolve anything that's in LayoutParser model zoo.
    """

    PREFIX = "lp://paddlepaddle/"

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
