<p align="left">
  <img src="https://github.com/Layout-Parser/layout-parser/raw/master/.github/layout-parser.png" alt="Layout Parser Logo" width="35%">
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

## Installation 

Use pip or conda to install the library:
```bash
pip install layoutparser

# Install Detectron2 for using DL Layout Detection Model
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.1.3#egg=detectron2' 

# Install the ocr components when necessary 
pip install layoutparser[ocr]      
```
This by default will install the CPU version of the Detectron2, and it should be able to run on most of the computers. But if you have a GPU, you can consider the GPU version of the Detectron2, referring to the [official instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

## Quick Start

We provide a series of examples for to help you start using the layout parser library: 

1. [Table OCR and Results Parsing](https://github.com/Layout-Parser/layout-parser/blob/master/examples/OCR%20Tables%20and%20Parse%20the%20Output.ipynb): `layoutparser` can be used for conveniently OCR documents and convert the output in to structured data. 

2. [Deep Layout Parsing Example](https://github.com/Layout-Parser/layout-parser/blob/master/examples/Deep%20Layout%20Parsing.ipynb): With the help of Deep Learning, `layoutparser` supports the analysis very complex documents and processing of the hierarchical structure in the layouts. 


## DL Assisted Layout Prediction Example 

![Example Usage](.github/example.png)

*The images shown in the figure above are: a screenshot of [this paper](https://arxiv.org/abs/2004.08686), an image from the [PRIMA Layout Analysis Dataset](https://www.primaresearch.org/dataset/), a screenshot of the [WSJ website](http://wsj.com), and an image from the [HJDataset](https://dell-research-harvard.github.io/HJDataset/).*

With only 4 lines of code in `layoutparse`, you can unlock the information from complex documents that existing tools could not provide. You can either choose a deep learning model from the [ModelZoo](https://github.com/Layout-Parser/layout-parser/blob/master/docs/notes/modelzoo.md), or load the model that you trained on your own. And use the following code to predict the layout as well as visualize it: 

```python
>>> import layoutparser as lp
>>> model = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config')
>>> layout = model.detect(image) # You need to load the image somewhere else, e.g., image = cv2.imread(...)
>>> lp.draw_box(image, layout,) # With extra configurations
```