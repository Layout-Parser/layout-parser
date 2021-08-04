from abc import ABC, abstractmethod
import os
import importlib


class BaseLayoutModel(ABC):
    
    @property
    @abstractmethod
    def DETECTOR_NAME(self):
        pass
    
    @abstractmethod
    def detect(self):
        pass

    # Add lazy loading mechanisms for layout models, refer to
    # layoutparser.ocr.BaseOCRAgent
    # TODO: Build a metaclass for lazy module loader
    @property
    @abstractmethod
    def DEPENDENCIES(self):
        """DEPENDENCIES lists all necessary dependencies for the class."""
        pass

    @property
    @abstractmethod
    def MODULES(self):
        """MODULES instructs how to import these necessary libraries."""
        pass

    @classmethod
    def _import_module(cls):
        for m in cls.MODULES:
            if importlib.util.find_spec(m["module_path"]):
                setattr(
                    cls, m["import_name"], importlib.import_module(m["module_path"])
                )
            else:
                raise ModuleNotFoundError(
                    f"\n "
                    f"\nPlease install the following libraries to support the class {cls.__name__}:"
                    f"\n    pip install {' '.join(cls.DEPENDENCIES)}"
                    f"\n "
                )

    def __new__(cls, *args, **kwargs):

        cls._import_module()
        return super().__new__(cls)