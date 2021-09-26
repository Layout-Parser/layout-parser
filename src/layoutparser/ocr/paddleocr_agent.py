# Copyright 2021 The Layout Parser team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import csv
import pickle
import numpy as np

import pandas as pd

from .base import BaseOCRAgent, BaseOCRElementType
from ..io import load_dataframe
from ..file_utils import is_paddleocr_available

if is_paddleocr_available():
    import paddleocr


class PaddleOCRAgent(BaseOCRAgent):
    """
    A wrapper for `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_ Text
    Detection APIs based on `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_.
    """

    DEPENDENCIES = ["paddleocr"]

    def __init__(self, languages="en", use_gpu=True, use_angle_cls=False, det=True, rec=True, cls=False, **kwargs):
        """Create a Tesseract OCR Agent.

        Args:
            languages (:obj:`list` or :obj:`str`, optional):
                You can specify the language code(s) of the documents to detect to improve
                accuracy. The supported language and their code can be found on
                `its github repo <https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/whl.md>`_.
                It supports llaguages：`ch`, `en`, `french`, `german`, `korean`, `japan`.
                Defaults to 'eng'.
        """
        self.lang = languages
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.configs = kwargs
        self.ocr = paddleocr.PaddleOCR(use_gpu=self.use_gpu, use_angle_cls=self.use_angle_cls, lang=self.lang)
   
    def resized_long(self, image, target=480):
        shape = image.shape
        if max(image.shape[0], image.shape[1]) >= target:
            return image
        if shape[0] >= shape[1]:
            ratio = 1.0 * target / shape[0]
            out = [int(shape[1] * ratio), target]
        else:
            ratio = 1.0 * target / shape[1]
            out = [target, int(shape[0] * ratio)]
        return cv2.resize(image, out)

    def pad_img_to_longer_edge(self, image):
        max_shape = max(image.shape[0], image.shape[1])
        out_img = np.ones([max_shape, max_shape, 3]) * 127
        out_img[:image.shape[0], :image.shape[1], :image.shape[2]] = image
        return out_img

    def detect(
        self, image, det=True, rec=True, cls=True,
        return_response=False, return_only_text=True
    ):
        """Send the input image for OCR.

        Args:
            image (:obj:`np.ndarray` or :obj:`str`):
                The input image array or the name of the image file
            det (:obj:`bool`, optional): 
                use text detection or not, if false, only rec will be exec.
                Default to `True`.
            rec (:obj:`bool`, optional): 
                Use text recognition or not, if false, only det will be exec. 
                Default to `True`.
            cls (:obj:`bool`, optional):
                Use 180 degree rotation text recognition or not. 
                Default to `True`.
            return_response (:obj:`bool`, optional):
                Whether directly return all output (string and boxes
                info) from Tesseract.
                Defaults to `False`.
            return_only_text (:obj:`bool`, optional):
                Whether return only the texts in the OCR results.
                Defaults to `False`.
        """ 
        image = self.resized_long(image)
        image = self.pad_img_to_longer_edge(image)
        res = self.ocr.ocr(image, det=det, rec=rec, cls=cls)

        if return_response:
            return res
        
        if return_only_text:
            return ['\n'.join(line[1][0] for line in res)]

        return ['\n'.join(line[1][0] for line in res)]

    @staticmethod
    def load_response(filename):
        with open(filename, "rb") as fp:
            res = pickle.load(fp)
        return res

    @staticmethod
    def save_response(res, file_name):

        with open(file_name, "wb") as fp:
            pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)