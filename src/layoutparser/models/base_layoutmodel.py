from typing import Optional, Tuple, Union, Dict
from abc import ABC, abstractmethod

from .model_config import LayoutModelConfig, add_identifier_for_config, layout_model_config_parser, is_lp_layout_model_config_any_format
from ..file_utils import requires_backends

class BaseLayoutModel(ABC):

    # TODO: Build a metaclass for lazy module loader
    @property
    @abstractmethod
    def DEPENDENCIES(self):
        """DEPENDENCIES lists all necessary dependencies for the class."""
        pass

    @property
    @abstractmethod
    def DETECTOR_NAME(self):
        pass

    @property
    @abstractmethod
    def MODEL_CATALOG(self) -> Dict[str, Dict[str, str]]:
        pass

    @abstractmethod
    def detect(self, image: Union["np.ndarray", "Image.Image"]):
        pass


    @abstractmethod
    def image_loader(self, image: Union["np.ndarray", "Image.Image"]):
        """It will process the input images appropriately to the target format."""
        pass
    
    def _parse_config(self, config_path:str, identifier:str) -> Union[LayoutModelConfig, str]:
        
        if is_lp_layout_model_config_any_format(config_path):
            config_path = add_identifier_for_config(config_path, identifier)
            for dataset_name in self.MODEL_CATALOG:
                if dataset_name in config_path:
                    default_model_arch = list(self.MODEL_CATALOG[dataset_name].keys())[0]
                    # Use the first model_name for the dataset as the default_model_arch
                    return layout_model_config_parser(config_path, self.DETECTOR_NAME, default_model_arch)
            raise ValueError(f"The config {config_path} is not a valid config for {self.__class__}")
        else:
            return config_path

    def config_parser(self, config_path:str, model_path: Optional[str], allow_empty_path=False) -> Tuple[str, str]:

        config_path = self._parse_config(config_path, "config")
        
        if isinstance(config_path, str) and model_path is None:
            if not allow_empty_path:
                raise ValueError(
                    f"Invalid config and model path pairs ({(config_path, model_path)}):"
                    f"When config_path is a regular URL, the model_path should not be empty"
                )
            else:
                return config_path, model_path
        elif isinstance(config_path, LayoutModelConfig) and model_path is None:
            model_path = config_path.dual()
        else:
            model_path = self._parse_config(model_path, "weight")

        config_path = config_path if isinstance(config_path, str) else config_path.full
        model_path = model_path if isinstance(model_path, str) else model_path.full
        return config_path, model_path

    def __new__(cls, *args, **kwargs):

        requires_backends(cls, cls.DEPENDENCIES)
        return super().__new__(cls)