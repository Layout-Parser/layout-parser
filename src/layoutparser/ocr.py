from abc import ABC, abstractmethod
from enum import IntEnum
import importlib, io

class BaseOCRElementType(IntEnum): 
    
    @property
    @abstractmethod
    def attr_name(self): pass
    

class BaseOCRAgent(ABC):
    
    @property
    @abstractmethod
    def DEPENDENCIES(self): 
        """DEPENDENCIES lists all necessary dependencies for the class.
        """
        pass
    
    @property
    @abstractmethod
    def MODULES(self): 
        """MODULES instructs how to import these necessary libraries. 
        
        Note: 
            Sometimes a python module have different installation name and module name (e.g., 
            `pip install tensorflow-gpu` when installing and `import tensorflow` when using
            ). And sometimes we only need to import a submodule but not whole module. MODULES 
            is designed for this purpose. 
        
        Returns:
            :obj: list(dict): A list of dict indicate how the model is imported. 
                
                Example::
                    
                    [{
                        "import_name": "_vision",
                        "module_path": "google.cloud.vision"
                    }]
                
                    is equivalent to self._vision = importlib.import_module("google.cloud.vision")
        """
        pass
    
    @classmethod
    def _import_module(cls):
        for m in cls.MODULES:
            if importlib.util.find_spec(m["module_path"]):
                setattr(cls, m["import_name"], importlib.import_module(m["module_path"]))
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
    
    @abstractmethod
    def detect(self, image): pass
