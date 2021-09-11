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

# Some code are adapted from
# https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py

from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
import sys
import os
import logging
import importlib.util
from types import ModuleType

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

###########################################
############ Layout Model Deps ############
###########################################

_torch_available = importlib.util.find_spec("torch") is not None
try:
    _torch_version = importlib_metadata.version("torch")
    logger.debug(f"PyTorch version {_torch_version} available.")
except importlib_metadata.PackageNotFoundError:
    _torch_available = False

_detectron2_available = importlib.util.find_spec("detectron2") is not None
try:
    _detectron2_version = importlib_metadata.version("detectron2")
    logger.debug(f"Detectron2 version {_detectron2_version} available")
except importlib_metadata.PackageNotFoundError:
    _detectron2_available = False

_paddle_available = importlib.util.find_spec("paddle") is not None
try:
    # The name of the paddlepaddle library:
    # Install name: pip install paddlepaddle
    # Import name: import paddle
    _paddle_version = importlib_metadata.version("paddlepaddle")
    logger.debug(f"Paddle version {_paddle_version} available.")
except importlib_metadata.PackageNotFoundError:
    _paddle_available = False

_effdet_available = importlib.util.find_spec("effdet") is not None
try:
    _effdet_version = importlib_metadata.version("effdet")
    logger.debug(f"Effdet version {_effdet_version} available.")
except importlib_metadata.PackageNotFoundError:
    _effdet_version = False

###########################################
############## OCR Tool Deps ##############
###########################################

_pytesseract_available = importlib.util.find_spec("pytesseract") is not None
try:
    _pytesseract_version = importlib_metadata.version("pytesseract")
    logger.debug(f"Pytesseract version {_pytesseract_version} available.")
except importlib_metadata.PackageNotFoundError:
    _pytesseract_available = False

try:
    _gcv_available = importlib.util.find_spec("google.cloud.vision") is not None
    try:
        _gcv_version = importlib_metadata.version(
            "google-cloud-vision"
        )  # This is slightly different
        logger.debug(f"Google Cloud Vision Utils version {_gcv_version} available.")
    except importlib_metadata.PackageNotFoundError:
        _gcv_available = False
except ModuleNotFoundError:
    _gcv_available = False


def is_torch_available():
    return _torch_available


def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def is_detectron2_available():
    return _detectron2_available


def is_paddle_available():
    return _paddle_available


def is_effdet_available():
    return _effdet_available


def is_pytesseract_available():
    return _pytesseract_available


def is_gcv_available():
    return _gcv_available


PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""

DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md and follow the ones
that match your environment. Typically the following would work for MacOS or Linux CPU machines:
    pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2' 
"""

PADDLE_IMPORT_ERROR = """
{0} requires the PaddlePaddle library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/PaddlePaddle/Paddle and follow the ones that match your environment.
"""

EFFDET_IMPORT_ERROR = """
{0} requires the effdet library but it was not found in your environment. You can install it with pip:
`pip install effdet`
"""

PYTESSERACT_IMPORT_ERROR = """
{0} requires the PyTesseract library but it was not found in your environment. You can install it with pip:
`pip install pytesseract`
"""

GCV_IMPORT_ERROR = """
{0} requires the Google Cloud Vision Python utils but it was not found in your environment. You can install it with pip:
`pip install google-cloud-vision==1`
"""

BACKENDS_MAPPING = dict(
    [
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("detectron2", (is_detectron2_available, DETECTRON2_IMPORT_ERROR)),
        ("paddle", (is_paddle_available, PADDLE_IMPORT_ERROR)),
        ("effdet", (is_effdet_available, EFFDET_IMPORT_ERROR)),
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        ("google-cloud-vision", (is_gcv_available, GCV_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not all(BACKENDS_MAPPING[backend][0]() for backend in backends):
        raise ImportError(
            "".join([BACKENDS_MAPPING[backend][1].format(name) for backend in backends])
        )


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Adapted from HuggingFace
    # https://github.com/huggingface/transformers/blob/c37573806ab3526dd805c49cbe2489ad4d68a9d7/src/transformers/file_utils.py#L1990

    def __init__(
        self, name, module_file, import_structure, module_spec=None, extra_objects=None
    ):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + sum(
            import_structure.values(), []
        )
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

        # Following [PEP 366](https://www.python.org/dev/peps/pep-0366/)
        # The __package__ variable should be set
        # https://docs.python.org/3/reference/import.html#__package__
        self.__package__ = self.__name__

    # Needed for autocompletion in an IDE
    def __dir__(self):
        return super().__dir__() + self.__all__

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        return importlib.import_module("." + module_name, self.__name__)

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
