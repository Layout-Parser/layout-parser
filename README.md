<p align="left">
  <img src=".github/layout-parser.png" alt="Layout Parser Logo" width="35%">
</p>

<p align="center">

[![Docs](https://readthedocs.org/projects/layout-parser/badge/)](https://layout-parser.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/layoutparser?color=%23099cec&label=PyPI%20package&logo=pypi&logoColor=white)](https://pypi.org/project/layoutparser/)
[![PyVersion](https://img.shields.io/pypi/pyversions/layoutparser?color=%23099cec&
)](https://pypi.org/project/layoutparser/)
[![License](https://img.shields.io/pypi/l/layoutparser)](https://github.com/Layout-Parser/layout-parser/blob/master/LICENSE)

</p>

---

Layout Parser is deep learning based tool for document image layout analysis tasks. 

## Quick Start 

1. Install the package
    ```bash
    pip install layoutparser

    # Install Detectron2 for using DL Layout Detection Model
    pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.1.3#egg=detectron2' 

    # Install the ocr components when necessary 
    pip install layoutparser[ocr]      
    ```

