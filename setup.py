from setuptools import setup, find_packages

setup(name         = "layoutparser",
      version      = "0.0.1",
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
        "pandas"
      ],
      include_package_data=True
      )