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

from iopath.common.file_io import HTTPURLHandler
from iopath.common.file_io import PathManager as PathManagerBase

# A trick learned from https://github.com/facebookresearch/detectron2/blob/65faeb4779e4c142484deeece18dc958c5c9ad18/detectron2/utils/file_io.py#L3


class DropboxHandler(HTTPURLHandler):
    """
    Supports download and file check for dropbox links
    """

    def _get_supported_prefixes(self):
        return ["https://www.dropbox.com"]

    def _isfile(self, path):
        return path in self.cache_map


PathManager = PathManagerBase()
PathManager.register_handler(DropboxHandler())