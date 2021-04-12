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

## How to Contribute?

This how-to-guide is abridged from the [MMOCR repo](https://github.com/open-mmlab/mmocr/blob/main/.github/CONTRIBUTING.md)

### Main Steps

1. Fork and pull the latest Layout Parser Repo
2. Checkout a new branch (do not use main branch for PRs)
3. Commit your changes
4. Create a PR

**Notes**:
1. If you plan to add some new features that involve large changes, please open an issue for discussion first.
2. If you are the author of some papers and would like to include your method to Layout Parser, please let us know (open an issue or contact the maintainers). We will much appreciate your contribution.
3. For new features and new modules, unit tests are required to improve the robustness.
4. You might want to run `pip install -r dev-requirements.txt` to install the dev-dependencies. 

### Code Style 

1. We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.
2. We use the following tools for linting and formatting:
    - pylint: linter
    - black: formatter
3. We suggest adding [type hints](https://docs.python.org/3/library/typing.html) for all APIs. 

Sincere thanks,

Zejiang (Shannon) Shen 
