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

from setuptools import setup, find_packages
import os 

# A trick from https://github.com/jina-ai/jina/blob/79b302c93b01689e82cf4b52f46522eb7497c404/setup.py#L20
pkg_name = 'layoutparser'
libinfo_py = os.path.join('src', pkg_name, '__init__.py')
libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
exec(version_line)  # gives __version__

setup(name         = "layoutparser",
      version      = __version__,
      author       = "Zejiang Shen, Ruochen Zhang, and Layout Parser Model Contributors",
      author_email = "layoutparser@gmail.com",
      license      = "Apache-2.0",
      url          = "https://github.com/Layout-Parser/layout-parser",
      package_dir  = {"": "src"},
      packages     = find_packages("src"),
      description  = "A unified toolkit for Deep Learning Based Document Image Analysis",
      long_description=open("README.md", "r", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      python_requires='>=3.6',
      install_requires=[
        "numpy", 
        "opencv-python",
        "scipy",
        "pandas",
        "pillow",
        "pyyaml>=5.1",
        "iopath",
        "pdfplumber",
        "pdf2image",
      ],
      extras_require={
        "ocr": [
          'google-cloud-vision==1',
          'pytesseract'
        ], 
        "gcv": [
          'google-cloud-vision==1',
        ],
        "tesseract": [
          'pytesseract'
        ],
        "layoutmodels": [
          "torch",
          "torchvision",
          "effdet"
        ],
        "effdet": [
          "torch",
          "torchvision",
          "effdet"
        ],
        "paddledetection": [
          "paddlepaddle==2.1.0"
        ],
      },
      include_package_data=True
      )