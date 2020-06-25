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

## DL Assisted Layout Prediction 

You can either choose a model from the [ModelZoo](https://github.com/Layout-Parser/layout-parser/docs/notes/modelzoo.md), or load your model. And use the following code to predict the layout as well as visualize it: 

```python
>>> import layoutparser as lp
>>> model = lp.Detectron2LayoutModel('lp://HJDataset/faster_rcnn_R_50_FPN_3x/config')
>>> layout = model.detect(image) # You need to load the image somewhere else, e.g., image = cv2.imread(...)
>>> lp.draw_box(image[...,::-1], layout, box_width=3)
```