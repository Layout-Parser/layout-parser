from typing import Union
from abc import ABC, abstractmethod

from ..file_utils import requires_backends


class BaseLayoutModel(ABC):
    @property
    @abstractmethod
    def DETECTOR_NAME(self):
        pass

    @abstractmethod
    def detect(self, image):
        pass

    @abstractmethod
    def image_loader(self, image: Union["np.ndarray", "Image.Image"]):
        """It will process the input images appropriately to the target format."""
        pass

    # Add lazy loading mechanisms for layout models, refer to
    # layoutparser.ocr.BaseOCRAgent
    # TODO: Build a metaclass for lazy module loader
    @property
    @abstractmethod
    def DEPENDENCIES(self):
        """DEPENDENCIES lists all necessary dependencies for the class."""
        pass

    def __new__(cls, *args, **kwargs):

        requires_backends(cls, cls.DEPENDENCIES)
        return super().__new__(cls)

    def _reconstruct_path_with_detector_name(self, path: str) -> str:
        """This function will add the detector name into the
        lp model config path to get the "canonical" model name.

        For example,
        for a given config_path `lp://HJDataset/faster_rcnn_R_50_FPN_3x/config`,it will
        transform it into `lp://<self.DETECTOR_NAME>/HJDataset/faster_rcnn_R_50_FPN_3x/config`.
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
                and self.DETECTOR_NAME not in model_name_segments
            ):
                return "lp://" + self.DETECTOR_NAME + "/" + path[len("lp://") :]
        return path
