from PIL import Image
import numpy as np
import torch

from .catalog import PathManager, LABEL_MAP_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout

__all__ = ["Detectron2LayoutModel"]


class Detectron2LayoutModel(BaseLayoutModel):
    """Create a Detectron2-based Layout Detection Model

    Args:
        config_path (:obj:`str`):
            The path to the configuration file.
        model_path (:obj:`str`, None):
            The path to the saved weights of the model.
            If set, overwrite the weights in the configuration file.
            Defaults to `None`.
        label_map (:obj:`dict`, optional):
            The map from the model prediction (ids) to real
            word labels (strings). If the config is from one of the supported
            datasets, Layout Parser will automatically initialize the label_map.
            Defaults to `None`.
        enforce_cpu(:obj:`bool`, optional):
            When set to `True`, it will enforce using cpu even if it is on a CUDA
            available device.
        extra_config (:obj:`list`, optional):
            Extra configuration passed to the Detectron2 model
            configuration. The argument will be used in the `merge_from_list
            <https://detectron2.readthedocs.io/modules/config.html
            #detectron2.config.CfgNode.merge_from_list>`_ function.
            Defaults to `[]`.

    Examples::
        >>> import layoutparser as lp
        >>> model = lp.models.Detectron2LayoutModel('lp://HJDataset/faster_rcnn_R_50_FPN_3x/config')
        >>> model.detect(image)

    """

    DEPENDENCIES = ["detectron2"]
    MODULES = [
        {
            "import_name": "_engine",
            "module_path": "detectron2.engine",
        },
        {"import_name": "_config", "module_path": "detectron2.config"},
    ]
    DETECTOR_NAME = "detectron2"

    def __init__(
        self,
        config_path,
        model_path=None,
        label_map=None,
        extra_config=[],
        enforce_cpu=False,
    ):

        if config_path.startswith("lp://") and label_map is None:
            dataset_name = config_path.lstrip("lp://").split("/")[0]
            label_map = LABEL_MAP_CATALOG[dataset_name]

        if enforce_cpu:
            extra_config.extend(["MODEL.DEVICE", "cpu"])

        cfg = self._config.get_cfg()
        config_path = self._reconstruct_path_with_detector_name(config_path)
        config_path = PathManager.get_local_path(config_path)
        cfg.merge_from_file(config_path)
        cfg.merge_from_list(extra_config)

        if model_path is not None:
            model_path = self._reconstruct_path_with_detector_name(model_path)
            cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg

        self.label_map = label_map
        self._create_model()

    def _reconstruct_path_with_detector_name(self, path: str) -> str:
        """This function will add the detector name (detectron2) into the
        lp model config path to get the "canonical" model name.

        For example, for a given config_path `lp://HJDataset/faster_rcnn_R_50_FPN_3x/config`,
        it will transform it into `lp://detectron2/HJDataset/faster_rcnn_R_50_FPN_3x/config`.
        However, if the config_path already contains the detector name, we won't change it.

        This function is a general step to support multiple backends in the layout-parser
        library.

        Args:
            path (str): The given input path that might or might not contain the detector name.

        Returns:
            str: a modified path that contains the detector name.
        """
        if path.startswith("lp://"):  # TODO: Move "lp://" to a constant
            model_name = path[len("lp://") :]
            model_name_segments = model_name.split("/")
            if (
                len(model_name_segments) == 3
                and "detectron2" not in model_name_segments
            ):
                return "lp://" + self.DETECTOR_NAME + "/" + path[len("lp://") :]
        return path

    def gather_output(self, outputs):

        instance_pred = outputs["instances"].to("cpu")

        layout = Layout()
        scores = instance_pred.scores.tolist()
        boxes = instance_pred.pred_boxes.tensor.tolist()
        labels = instance_pred.pred_classes.tolist()

        for score, box, label in zip(scores, boxes, labels):
            x_1, y_1, x_2, y_2 = box

            if self.label_map is not None:
                label = self.label_map.get(label, label)

            cur_block = TextBlock(
                Rectangle(x_1, y_1, x_2, y_2), type=label, score=score
            )
            layout.append(cur_block)

        return layout

    def _create_model(self):
        self.model = self._engine.DefaultPredictor(self.cfg)

    def detect(self, image):
        """Detect the layout of a given image.

        Args:
            image (:obj:`np.ndarray` or `PIL.Image`): The input image to detect.

        Returns:
            :obj:`~layoutparser.Layout`: The detected layout of the input image
        """

        # Convert PIL Image Input
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image)

        outputs = self.model(image)
        layout = self.gather_output(outputs)
        return layout
