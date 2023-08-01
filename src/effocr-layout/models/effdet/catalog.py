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

from iopath.common.file_io import PathHandler

from ..base_catalog import PathManager

MODEL_CATALOG = {
    "PubLayNet": {
        "tf_efficientdet_d0": "https://www.dropbox.com/s/ukbw5s673633hsw/publaynet-tf_efficientdet_d0.pth.tar?dl=1",
        "tf_efficientdet_d1": "https://www.dropbox.com/s/gxy11xkkiwnpgog/publaynet-tf_efficientdet_d1.pth.tar?dl=1"
    },
    "MFD": {
        "tf_efficientdet_d0": "https://www.dropbox.com/s/dkr22iux7thlhel/mfd-tf_efficientdet_d0.pth.tar?dl=1",
        "tf_efficientdet_d1": "https://www.dropbox.com/s/icmbiaqr5s9bz1x/mfd-tf_efficientdet_d1.pth.tar?dl=1"
    }
}

# In effdet training scripts, it requires the label_map starting
# from 1 instead of 0
LABEL_MAP_CATALOG = {
    "PubLayNet": {
        1: "Text", 
        2: "Title", 
        3: "List", 
        4: "Table", 
        5: "Figure"
    },
    "MFD": {
        1: "Equation",
    }
}

class LayoutParserEfficientDetModelHandler(PathHandler):
    """
    Resolve anything that's in LayoutParser model zoo.
    """

    PREFIX = "lp://efficientdet/"

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


PathManager.register_handler(LayoutParserEfficientDetModelHandler())
