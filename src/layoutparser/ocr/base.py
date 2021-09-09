from abc import ABC, abstractmethod
from enum import IntEnum

from ..file_utils import requires_backends

class BaseOCRElementType(IntEnum):
    @property
    @abstractmethod
    def attr_name(self):
        pass


class BaseOCRAgent(ABC):
    @property
    @abstractmethod
    def DEPENDENCIES(self):
        """DEPENDENCIES lists all necessary dependencies for the class."""
        pass

    def __new__(cls, *args, **kwargs):

        requires_backends(cls, cls.DEPENDENCIES)
        return super().__new__(cls)

    @abstractmethod
    def detect(self, image):
        pass
