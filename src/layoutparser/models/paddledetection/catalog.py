# Copyright 2021 The Layout Parser team and Paddle Detection model 
# contributors. All rights reserved.
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

import os
import logging
from typing import Any, Optional
from urllib.parse import urlparse
import tarfile
import uuid

from iopath.common.file_io import PathHandler
from iopath.common.file_io import HTTPURLHandler
from iopath.common.file_io import get_cache_dir, file_lock
from iopath.common.download import download

from ..base_catalog import PathManager

MODEL_CATALOG = {
    "PubLayNet": {
        "ppyolov2_r50vd_dcn_365e": "https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar",
    },
    "TableBank": {
        "ppyolov2_r50vd_dcn_365e": "https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_word.tar",
        # "ppyolov2_r50vd_dcn_365e_tableBank_latex": "https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_tableBank_latex.tar",
        # TODO: Train a single tablebank model for paddlepaddle
    },
}

# fmt: off
LABEL_MAP_CATALOG = {
    "PubLayNet": {
        0: "Text",
        1: "Title",
        2: "List",
        3: "Table",
        4: "Figure"},
    "TableBank": {
        0: "Table"
    },
}
# fmt: on


# Paddle model package everything in tar files, and each model's tar file should contain
# the following files in the list:
_TAR_FILE_NAME_LIST = [
    "inference.pdiparams",
    "inference.pdiparams.info",
    "inference.pdmodel",
]


def _get_untar_directory(tar_file: str) -> str:

    base_path = os.path.dirname(tar_file)
    file_name = os.path.splitext(os.path.basename(tar_file))[0]
    target_folder = os.path.join(base_path, file_name)

    return target_folder


def _untar_model_weights(model_tar):
    """untar model files"""

    model_dir = _get_untar_directory(model_tar)

    if not os.path.exists(
        os.path.join(model_dir, _TAR_FILE_NAME_LIST[0])
    ) or not os.path.exists(os.path.join(model_dir, _TAR_FILE_NAME_LIST[2])):
        # the path to save the decompressed file
        os.makedirs(model_dir, exist_ok=True)
        with tarfile.open(model_tar, "r") as tarobj:
            for member in tarobj.getmembers():
                filename = None
                for tar_file_name in _TAR_FILE_NAME_LIST:
                    if tar_file_name in member.name:
                        filename = tar_file_name
                if filename is None:
                    continue
                file = tarobj.extractfile(member)
                with open(os.path.join(model_dir, filename), "wb") as model_file:
                    model_file.write(file.read())
    return model_dir


def is_cached_folder_exists_and_valid(cached):
    possible_extracted_model_folder = _get_untar_directory(cached)
    if not os.path.exists(possible_extracted_model_folder):
        return False
    for tar_file in _TAR_FILE_NAME_LIST:
        if not os.path.exists(os.path.join(possible_extracted_model_folder, tar_file)):
            return False
    return True


class PaddleModelURLHandler(HTTPURLHandler):
    """
    Supports download and file check for Baidu Cloud links
    """

    MAX_FILENAME_LEN = 250

    def _get_supported_prefixes(self):
        return ["https://paddle-model-ecology.bj.bcebos.com"]

    def _isfile(self, path):
        return path in self.cache_map

    def _get_local_path(
        self,
        path: str,
        force: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        As paddle model stores all files in tar files, we need to extract them
        and get the newly extracted folder path. This function rewrites the base
        function to support the following situations:

        1. If the tar file is not downloaded, it will download the tar file,
            extract it to the target folder, delete the downloaded tar file,
            and return the folder path.
        2. If the extracted target folder is present, and all the necessary model
            files are present (specified in _TAR_FILE_NAME_LIST), it will
            return the folder path.
        3. If the tar file is downloaded, but the extracted target folder is not
            present (or it doesn't contain the necessary files in _TAR_FILE_NAME_LIST),
            it will extract the tar file to the target folder, delete the tar file,
            and return the folder path.

        """
        self._check_kwargs(kwargs)
        if (
            force
            or path not in self.cache_map
            or not os.path.exists(self.cache_map[path])
        ):
            logger = logging.getLogger(__name__)
            parsed_url = urlparse(path)
            dirname = os.path.join(
                get_cache_dir(cache_dir), os.path.dirname(parsed_url.path.lstrip("/"))
            )
            filename = path.split("/")[-1]
            if len(filename) > self.MAX_FILENAME_LEN:
                filename = filename[:100] + "_" + uuid.uuid4().hex

            cached = os.path.join(dirname, filename)

            if is_cached_folder_exists_and_valid(cached):
                # When the cached folder exists and valid, we don't need to redownload
                # the tar file.
                self.cache_map[path] = _get_untar_directory(cached)

            else:
                with file_lock(cached):
                    if not os.path.isfile(cached):
                        logger.info("Downloading {} ...".format(path))
                        cached = download(path, dirname, filename=filename)

                    if path.endswith(".tar"):
                        model_dir = _untar_model_weights(cached)
                        try:
                            os.remove(cached)  # remove the redundant tar file
                            # TODO: remove the .lock file .
                        except:
                            logger.warning(
                                f"Not able to remove the cached tar file {cached}"
                            )

                logger.info("URL {} cached in {}".format(path, model_dir))
                self.cache_map[path] = model_dir

        return self.cache_map[path]


class LayoutParserPaddleModelHandler(PathHandler):
    """
    Resolve anything that's in LayoutParser model zoo.
    """

    PREFIX = "lp://paddledetection/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        model_name = path[len(self.PREFIX) :]
        dataset_name, *model_name, data_type = model_name.split("/")

        if data_type == "weight":
            model_url = MODEL_CATALOG[dataset_name]["/".join(model_name)]
        else:
            raise ValueError(f"Unknown data_type {data_type}")
        return PathManager.get_local_path(model_url, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(PaddleModelURLHandler())
PathManager.register_handler(LayoutParserPaddleModelHandler())
