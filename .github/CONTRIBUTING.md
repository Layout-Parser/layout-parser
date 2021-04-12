# Contributing to Layout Parser

🎉👍 Thank you for reading this and plan to contribute! We hope you can join us and work on
this exciting project that can transform document image analysis pipelines with the full
power of Deep Learning.

All kinds of contributions are welcome, including but not limited to:

- Better documentation and examples for more use cases
- New pre-trained layout detection models
- New features

## Planned features 

We are planning to improve different aspects of Layout Parser, any feedbacks and contributions are welcomed! 

### Layout Modeling

(Pre-trained) layout models are one of the most components in Layout Parser, and we are planning to broadening the support for layout models: 

- Support frameworks other than Detectron2, e.g., [MMOCR](https://github.com/open-mmlab/mmocr). It may leads to easier installation and support for more application scenarios like recipt or invoice detection. 
- Support segmentation-based models, e.g., [dhSegment](https://github.com/dhlab-epfl/dhSegment)
- Better customized training of layout detection models, see [layout-model-training](https://github.com/Layout-Parser/layout-model-training)
- Reproducing novel layout models in the current framework, e.g., [CascadeTabNet](https://github.com/DevashishPrasad/CascadeTabNet)

We are also working on the Layout Parser platforms that can support users' sharing their own models. Please check more details in [community-platform](https://github.com/Layout-Parser/community-platform).

### Advanced Layout Pipeline

- Support defining `Pipeline` that specifies an end-to-end layout processing pipeline for complex documents. 

### Command Line Tool and Layout Detection Service

Layout Parser can be easily turned into a command line tool or service to process documents in bulk. 

- Build a command line tool based on `Click` that supports commands like `layoutparser process --path <path/to/document/folders>`
- Build a RESTful Layout Parser service based on tools like `FastAPI` with similar supports as the ccommand line tool. 
- Performance improvements for these service 

### Easy Installation and Deployment 

- Better ways for installing Detectron2 and related components on Windows machines 
- A Docker configuration for installing the Layout Parser

Sincere thanks,

Zejiang (Shannon) Shen 
