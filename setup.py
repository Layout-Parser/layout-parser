from setuptools import setup, find_packages

setup(name="layoutparser",
      version="0.0.1",
      author="Zejiang Shen",
      license="Apache",
      url="https://github.com/Layout-Parser/layout-parser",
      package_dir={"": "src"},
      packages=find_packages("src"),
      install_requires=[
        "numpy", 
        "opencv-python"
      ]
      )