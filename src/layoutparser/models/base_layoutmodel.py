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
    def image_loader(self, image: Union["ndarray", "Image"]):
        """It will process the input images appropriately to the target format. 
        """
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