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
      author       = "Zejiang Shen Ruochen Zhang",
      license      = "Apache-2.0",
      url          = "https://github.com/Layout-Parser/layout-parser",
      package_dir  = {"": "src"},
      packages     = find_packages("src"),
      long_description=open("README.md", "r", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      python_requires='>=3.6',
      install_requires=[
        "numpy", 
        "opencv-python",
        "pandas",
        "pillow",
        "pyyaml>=5.1",
        "iopath",
      ],
      extras_require={
        "ocr": [
          'google-cloud-vision==1',
          'pytesseract'
        ], 
        "effdet": [
          "torch",
          "torchvision",
          "effdet"
        ],
        "detectron2": [
          "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.4#egg=detectron2"
        ],
        "paddledetection": [
          "paddlepaddle==2.1.0"
        ]
      },
      include_package_data=True
      )