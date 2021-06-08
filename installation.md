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

## [Optional] Install PaddleDetection for Using Layout Models

### For Windows and Linux Users 

If you would like to use PaddleDetection deep learning models for layout detection, you also need to install paddle on your computer. This could be done by running the following command: 

```bash
# CPU version
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

If you would like to use PaddlePaddle GPU version to predict, you need to uninstall paddlepaddle first, and running the following conmand:

```bash
# If you already have installed the CPU version paddle
pip uninstall paddlepaddle
# GPU version
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

### For Mac OS users

This could be done by running the following command: 

```bash
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

- For more CUDA version or environment to quick install, please refer to the [PaddlePaddle Quick Installation document](https://www.paddlepaddle.org.cn/install/quick)
- For more installation methods such as conda or compile with source code, please refer to the [installation document](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)

Please make sure that your PaddlePaddle is installed successfully and the version is not lower than the required version. Use the following command to verify.

```bash
# check
>>> import paddle
>>> paddle.utils.run_check()

# confirm the paddle's version
python -c "import paddle; print(paddle.__version__)"
```

## [Optional] Install OCR utils

Layout Parser also comes with supports for OCR functions. In order to use them, you need to install the PaddleOCR utils via: 

```bash
pip3 install -U layoutparser[ocr]
```

Additionally, if you want to use the Tesseract-OCR engine, you also need to install it on your computer. Please check the 
[official documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html) for detailed installation instructions. 

## [Optional] Install PaddleOCR utils

Layout Parser also comes with supports for OCR functions. In order to use them, you need to install the OCR utils via: 

```bash
pip3 install -U layoutparser[paddleocr]
```

. Please check the [official documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/whl.md) for detailed installation instructions. 

## Known issues

<details><summary>Error: instantiating `lp.GCVAgent.with_credential` returns module 'google.cloud.vision' has no attribute 'types'. </summary>
<p>

In this case, you have a newer version of the google-cloud-vision. Please consider downgrading the API using: 
```bash
pip install layoutparser[ocr]
```
</p>
</details>