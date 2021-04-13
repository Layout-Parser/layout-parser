# Installation

## Install Python

Layout Parser is a Python package that requires Python >= 3.6. If you do not have Python installed on your computer, you might want to turn to [the official instruction](https://www.python.org/downloads/) to download and install the appropriate version of Python.

## Install the Layout Parser main library

Installing the Layout Parser library is very straightforward: you just need to run the following command: 

```bash
pip3 install -U layoutparser
```

## [Optional] Install Detectron2 for Using Layout Models

### For Mac OS and Linux Users 

If you would like to use deep learning models for layout detection, you also need to install Detectron2 on your computer. This could be done by running the following command: 

```bash
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2' 
```

This might take some time as the command will *compile* the library. You might also want to install a Detectron2 version 
with GPU support or encounter some issues during the installation process. Please refer to the official Detectron2 
[installation instruction](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for detailed
information. 

### For Windows users

As reported by many users, the installation of Detectron2 can be rather tricky on Windows platforms. In our extensive tests, we find that it is nearly impossible to provide a one-line installation command for Windows users. As a workaround solution, for now we list the possible challenges for installing Detectron2 on Windows, and attach helpful resources for solving them. We are also investigating other possibilities to avoid installing Detectron2 to use pre-trained models. If you have any suggestions or ideas, please feel free to [submit an issue](https://github.com/Layout-Parser/layout-parser/issues) in our repo. 

1. Challenges for installing `pycocotools` 
    - You can find detailed instructions on [this post](https://changhsinlee.com/pycocotools/) from Chang Hsin Lee. 
    - Another solution is try to install `pycocotools-windows`, see https://github.com/cocodataset/cocoapi/issues/415. 
2. Challenges for installing `Detectron2` 
    - [@ivanpp](https://github.com/ivanpp) curates a detailed description for installing `Detectron2` on Windows: [Detectron2 walkthrough (Windows)](https://ivanpp.cc/detectron2-walkthrough-windows/#step3installdetectron2)
    - `Detectron2` maintainers claim that they won't provide official support for Windows (see [1](https://github.com/facebookresearch/detectron2/issues/9#issuecomment-540974288) and [2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)), but Detectron2 is continuously built on windows with CircleCI (see [3](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#common-installation-issues)). Hopefully this situation will be improved in the future.


## [Optional] Install OCR utils

Layout Parser also comes with supports for OCR functions. In order to use them, you need to install the OCR utils via: 

```bash
pip3 install -U layoutparser[ocr]
```

Additionally, if you want to use the Tesseract-OCR engine, you also need to install it on your computer. Please check the 
[official documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html) for detailed installation instructions. 

## Known issues

<details><summary>Error: instantiating `lp.GCVAgent.with_credential` returns module 'google.cloud.vision' has no attribute 'types'. </summary>
<p>

In this case, you have a newer version of the google-cloud-vision. Please consider downgrading the API using: 
```bash
pip install layoutparser[ocr]
```
</p>
</details>