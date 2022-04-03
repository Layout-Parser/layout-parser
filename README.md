<p align="center">
  <img src="https://github.com/Layout-Parser/layout-parser/raw/main/.github/layout-parser.png" alt="Layout Parser Logo" width="35%">
  <h3 align="center">
  A unified toolkit for Deep Learning Based Document Image Analysis
  </h3>
</p>

<p align=center>
<a href="https://pypi.org/project/layoutparser/"><img src="https://img.shields.io/pypi/v/layoutparser?color=%23099cec&label=PyPI%20package&logo=pypi&logoColor=white" title="The current version of Layout Parser"></a>
<a href="https://github.com/Layout-Parser/layout-parser/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/layoutparser" title="Layout Parser uses Apache 2 License"></a>
<img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/layoutparser">
</p>

<p align=center>
<a href="https://arxiv.org/abs/2103.15348"><img src="https://img.shields.io/badge/paper-2103.15348-b31b1b.svg" title="Layout Parser Paper"></a>
<a href="https://layout-parser.github.io"><img src="https://img.shields.io/badge/website-layout--parser.github.io-informational.svg" title="Layout Parser Paper"></a>
<a href="https://layout-parser.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/doc-layout--parser.readthedocs.io-light.svg" title="Layout Parser Documentation"></a>
</p>

---

## What is LayoutParser

![Example Usage](https://github.com/Layout-Parser/layout-parser/raw/main/.github/example.png)

LayoutParser aims to provide a wide range of tools that aims to streamline Document Image Analysis (DIA) tasks. Please check the LayoutParser [demo video](https://youtu.be/8yA5xB4Dg8c) (1 min) or [full talk](https://www.youtube.com/watch?v=YG0qepPgyGY) (15 min) for details. And here are some key features:

- LayoutParser provides a rich repository of deep learning models for layout detection as well as a set of unified APIs for using them. For example, 
  
  <details>
  <summary>Perform DL layout detection in 4 lines of code</summary>
  
  ```python
  import layoutparser as lp
  model = lp.AutoLayoutModel('lp://EfficientDete/PubLayNet')
  # image = Image.open("path/to/image")
  layout = model.detect(image) 
  ```
  
  </details>

- LayoutParser comes with a set of layout data structures with carefully designed APIs that are optimized for document image analysis tasks. For example, 

  <details>
  <summary>Selecting layout/textual elements in the left column of a page</summary>
  
  ```python
  image_width = image.size[0]
  left_column = lp.Interval(0, image_width/2, axis='x')
  layout.filter_by(left_column, center=True) # select objects in the left column 
  ```
  
  </details>

  <details>
  <summary>Performing OCR for each detected Layout Region</summary>
  
  ```python
  ocr_agent = lp.TesseractAgent()
  for layout_region in layout: 
      image_segment = layout_region.crop(image)
      text = ocr_agent.detect(image_segment)
  ```
  
  </details>  
    
  <details>
  <summary>Flexible APIs for visualizing the detected layouts</summary>
  
  ```python
  lp.draw_box(image, layout, box_width=1, show_element_id=True, box_alpha=0.25)
  ```
  
  </details>  
    
  </details>  
    
  <details>
  <summary>Loading layout data stored in json, csv, and even PDFs</summary>
  
  ```python 
  layout = lp.load_json("path/to/json")
  layout = lp.load_csv("path/to/csv")
  pdf_layout = lp.load_pdf("path/to/pdf")
  ```
  
  </details>

- LayoutParser is also a open platform that enables the sharing of layout detection models and DIA pipelines among the community. 
  <details>
  <summary><a href="https://layout-parser.github.io/platform/">Check</a> the LayoutParser open platform</summary>
  </details>

  <details>
  <summary><a href="https://github.com/Layout-Parser/platform">Submit</a> your models/pipelines to LayoutParser</summary>
  </details>

## Installation 

After several major updates, layoutparser provides various functionalities and deep learning models from different backends. But it still easy to install layoutparser, and we designed the installation method in a way such that you can choose to install only the needed dependencies for your project:

```bash
pip install layoutparser # Install the base layoutparser library with  
pip install "layoutparser[layoutmodels]" # Install DL layout model toolkit 
pip install "layoutparser[ocr]" # Install OCR toolkit
```

Extra steps are needed if you want to use Detectron2-based models. Please check [installation.md](installation.md) for additional details on layoutparser installation. 

## Examples 

We provide a series of examples for to help you start using the layout parser library: 

1. [Table OCR and Results Parsing](https://github.com/Layout-Parser/layout-parser/blob/main/examples/OCR%20Tables%20and%20Parse%20the%20Output.ipynb): `layoutparser` can be used for conveniently OCR documents and convert the output in to structured data. 

2. [Deep Layout Parsing Example](https://github.com/Layout-Parser/layout-parser/blob/main/examples/Deep%20Layout%20Parsing.ipynb): With the help of Deep Learning, `layoutparser` supports the analysis very complex documents and processing of the hierarchical structure in the layouts. 

## Contributing

We encourage you to contribute to Layout Parser! Please check out the [Contributing guidelines](.github/CONTRIBUTING.md) for guidelines about how to proceed. Join us!

## Citing `layoutparser`

If you find `layoutparser` helpful to your work, please consider citing our tool and [paper](https://arxiv.org/pdf/2103.15348.pdf) using the following BibTeX entry.

```
@article{shen2021layoutparser,
  title={LayoutParser: A Unified Toolkit for Deep Learning Based Document Image Analysis},
  author={Shen, Zejiang and Zhang, Ruochen and Dell, Melissa and Lee, Benjamin Charles Germain and Carlson, Jacob and Li, Weining},
  journal={arXiv preprint arXiv:2103.15348},
  year={2021}
}
```