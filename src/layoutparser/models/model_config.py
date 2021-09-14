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

"""
Inside layoutparser, we support the following formats for specifying layout model configs 
or weights:

1. URL-based formats:
    - A local path: ~/models/publaynet/path
    - Link to the models: https://web/url/to/models

2. LayoutParser Based Model/Config Path Formats:
    - Full format: lp://<backend-name>/<dataset-name>/<model-architecture-name>
    - Short format: lp://<dataset-name>/<model-architecture-name> 
    - Brief format: lp://<dataset-name>

For each LayoutParser-based format, you could also add a `config` or `weight` identifier 
after them: 
    - Full format: lp://<backend-name>/<dataset-name>/<model-architecture-name>/<config, weight>
    - Short format: lp://<dataset-name>/<model-architecture-name>/<config, weight> 
    - Brief format: lp://<dataset-name>/<config, weight>
"""

from dataclasses import dataclass

LAYOUT_PARSER_MODEL_PREFIX = "lp://"
ALLOWED_LAYOUT_MODEL_IDENTIFIER_NAMES = ["config", "weight"]


@dataclass
class LayoutModelConfig:

    backend_name: str
    dataset_name: str
    model_arch: str
    identifier: str

    def __post_init__(self):
        assert self.identifier in ALLOWED_LAYOUT_MODEL_IDENTIFIER_NAMES

    @property
    def full(self):
        return LAYOUT_PARSER_MODEL_PREFIX + "/".join(
            [self.backend_name, self.dataset_name, self.model_arch, self.identifier]
        )

    @property
    def short(self):
        return LAYOUT_PARSER_MODEL_PREFIX + "/".join(
            [self.dataset_name, self.model_arch, self.identifier]
        )

    @property
    def brief(self):
        return LAYOUT_PARSER_MODEL_PREFIX + "/".join([self.dataset_name, self.model_arch])

    def dual(self):
        for identifier in ALLOWED_LAYOUT_MODEL_IDENTIFIER_NAMES:
            if identifier != self.identifier:
                break

        return self.__class__(
            backend_name=self.backend_name,
            dataset_name=self.dataset_name,
            model_arch=self.model_arch,
            identifier=identifier,
        )


def is_lp_layout_model_config_any_format(config: str) -> bool:
    if not config.startswith(LAYOUT_PARSER_MODEL_PREFIX):
        return False
    if len(config[len(LAYOUT_PARSER_MODEL_PREFIX) :].split("/")) not in [1, 2, 3, 4]:
        return False
    return True


def add_identifier_for_config(config: str, identifier: str) -> str:
    return config.rstrip("/").rstrip(f"/{identifier}") + f"/{identifier}"


def layout_model_config_parser(
    config, backend_name=None, model_arch=None
) -> LayoutModelConfig:

    assert config.split("/")[-1] in ALLOWED_LAYOUT_MODEL_IDENTIFIER_NAMES, (
        f"The input config {config} does not contain identifier information."
        f"Consider run `config = add_identifier_for_config(config, identifier)` first."
    )

    parts = config[len(LAYOUT_PARSER_MODEL_PREFIX) :].split("/")
    if len(parts) == 4:  # Full format
        backend_name, dataset_name, model_arch, identifier = parts
    elif len(parts) == 3:  # Short format
        assert backend_name != None
        
        if parts[0] == backend_name:
            # lp://<backend-name>/<dataset-name>/<identifier>
            assert model_arch != None
            _, dataset_name, identifier = parts
        else:
            # lp://<dataset-name>/<model-arch>/<identifier>
            dataset_name, model_arch, identifier = parts
            
    elif len(parts) == 2:  # brief format
        assert backend_name != None
        assert model_arch != None
        if parts[0] == backend_name:
            # lp://<backend-name>/<identifier>    
            raise ValueError(f"Invalid LP Model Config {config}")
            
        # lp://<dataset-name>/<identifier>    
        dataset_name, identifier = parts
    else:
        raise ValueError(f"Invalid LP Model Config {config}")

    return LayoutModelConfig(
        backend_name=backend_name,
        dataset_name=dataset_name,
        model_arch=model_arch,
        identifier=identifier,
    )
