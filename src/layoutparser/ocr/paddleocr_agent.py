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

import pickle
import cv2
import numpy as np

from .base import BaseOCRAgent
from ..file_utils import is_paddleocr_available

if is_paddleocr_available():
    import paddleocr


class PaddleOCRAgent(BaseOCRAgent):
    """
    A wrapper for `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_ Text
    Detection APIs based on `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_.
    """

    DEPENDENCIES = ["paddleocr"]

    def __init__(self, languages="en", use_gpu=False, use_angle_cls=False):
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
        self.ocr = paddleocr.PaddleOCR(use_gpu=self.use_gpu, use_angle_cls=self.use_angle_cls, lang=self.lang)

    def resized_long(self, image, target_size):
        shape = image.shape
        if max(image.shape[0], image.shape[1]) >= target_size:
            return image
        if shape[0] >= shape[1]:
            ratio = 1.0 * target_size / shape[0]
            out = (int(shape[1] * ratio), target_size)
        else:
            ratio = 1.0 * target_size / shape[1]
            out = (target_size, int(shape[0] * ratio))
        return cv2.resize(image, out)

    def pad_img_to_longer_edge(self, image, padding_value=127):
        max_shape = max(image.shape[0], image.shape[1])
        out_img = np.ones([max_shape, max_shape, 3]) * padding_value
        out_img[:image.shape[0], :image.shape[1], :image.shape[2]] = image
        return out_img

    def _detect(self, img_content, target_size, padding_value,
            det, rec, cls, threshold):
        res = {}
        img_content = self.resized_long(img_content, target_size)
        img_content = self.pad_img_to_longer_edge(img_content, padding_value)
        result = self.ocr.ocr(img_content, det=det, rec=rec, cls=cls)
        text = []
        for line in result:
            if line[1][1]>threshold:
                text.append(line[1][0])
        res["text"] = '\n'.join(text)
        return res

    def detect(
        self, image, target_size=480, padding_value=127,
        det=True, rec=True, cls=True, threshold=0.5,
        return_response=False, return_only_text=True
    ):
        """Send the input image for OCR.

        Args:
            image (:obj:`np.ndarray` or :obj:`str`):
                The input image array or the name of the image file
            target_size (:obj:`int`, optional):
                The size of the longest side after resize.
                Default to `480`.
            padding_value (:obj:`int`, optional):
                The padding value will apply to get a square image.
                Default to `127`.
            det (:obj:`bool`, optional):
                use text detection or not, if false, only rec will be exec.
                Default to `True`.
            rec (:obj:`bool`, optional):
                Use text recognition or not, if false, only det will be exec.
                Default to `True`.
            cls (:obj:`bool`, optional):
                Use 180 degree rotation text recognition or not.
                Default to `True`.
            threshold (:obj:`float`, optional):
                Filter the recognition results with recognition scores less than threshold.
                Default to '0.5'.
            return_response (:obj:`bool`, optional):
                Whether directly return all output (string and boxes
                info) from Tesseract.
                Defaults to `False`.
            return_only_text (:obj:`bool`, optional):
                Whether return only the texts in the OCR results.
                Defaults to `False`.
        """

        res = self._detect(image, target_size, padding_value, det, rec, cls, threshold)

        if return_response:
            return res

        if return_only_text:
            return res["text"]

        return res["text"]

    @staticmethod
    def load_response(filename):
        with open(filename, "rb") as fp:
            res = pickle.load(fp)
        return res

    @staticmethod
    def save_response(res, file_name):

        with open(file_name, "wb") as fp:
            pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)
