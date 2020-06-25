Quickstart
================================


Installation
--------------------------------

Use pip or conda to install the library:

.. code-block:: bash

    pip install layoutparser

    # Install Detectron2 for using DL Layout Detection Model
    pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.1.3#egg=detectron2' 

    # Install the ocr components when necessary 
    pip install layoutparser[ocr]      

This by default will install the CPU version of the Detectron2, and it should be able to run on most of the computers. But if you have a GPU, you can consider the GPU version of the Detectron2, referring to the `official instructions <https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md>`_.