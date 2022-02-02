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

import pandas as pd

from .base import BaseOCRAgent, BaseOCRElementType
from ..io import load_dataframe
from ..file_utils import is_pytesseract_available

if is_pytesseract_available():
    import pytesseract


class TesseractFeatureType(BaseOCRElementType):
    """
    The element types for Tesseract Detection API
    """

    PAGE = 0
    BLOCK = 1
    PARA = 2
    LINE = 3
    WORD = 4

    @property
    def attr_name(self):
        name_cvt = {
            TesseractFeatureType.PAGE: "page_num",
            TesseractFeatureType.BLOCK: "block_num",
            TesseractFeatureType.PARA: "par_num",
            TesseractFeatureType.LINE: "line_num",
            TesseractFeatureType.WORD: "word_num",
        }
        return name_cvt[self]

    @property
    def group_levels(self):
        levels = ["page_num", "block_num", "par_num", "line_num", "word_num"]
        return levels[: self + 1]


class TesseractAgent(BaseOCRAgent):
    """
    A wrapper for `Tesseract <https://github.com/tesseract-ocr/tesseract>`_ Text
    Detection APIs based on `PyTesseract <https://github.com/tesseract-ocr/tesseract>`_.
    """

    DEPENDENCIES = ["pytesseract"]

    def __init__(self, languages="eng", **kwargs):
        """Create a Tesseract OCR Agent.

        Args:
            languages (:obj:`list` or :obj:`str`, optional):
                You can specify the language code(s) of the documents to detect to improve
                accuracy. The supported language and their code can be found on
                `its github repo <https://github.com/tesseract-ocr/langdata>`_.
                It supports two formats: 1) you can pass in the languages code as a string
                of format like `"eng+fra"`, or 2) you can pack them as a list of strings
                `["eng", "fra"]`.
                Defaults to 'eng'.
        """
        self.lang = languages if isinstance(languages, str) else "+".join(languages)
        self.configs = kwargs

    @classmethod
    def with_tesseract_executable(cls, tesseract_cmd_path, **kwargs):

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
        return cls(**kwargs)

    def _detect(self, img_content):
        res = {}
        res["text"] = pytesseract.image_to_string(
            img_content, lang=self.lang, **self.configs
        )
        _data = pytesseract.image_to_data(img_content, lang=self.lang, **self.configs)
        res["data"] = pd.read_csv(
            io.StringIO(_data),
            quoting=csv.QUOTE_NONE,
            encoding="utf-8",
            sep="\t",
            converters={"text": str},
        )
        return res

    def detect(
        self, image, return_response=False, return_only_text=True, agg_output_level=None
    ):
        """Send the input image for OCR.

        Args:
            image (:obj:`np.ndarray` or :obj:`str`):
                The input image array or the name of the image file
            return_response (:obj:`bool`, optional):
                Whether directly return all output (string and boxes
                info) from Tesseract.
                Defaults to `False`.
            return_only_text (:obj:`bool`, optional):
                Whether return only the texts in the OCR results.
                Defaults to `False`.
            agg_output_level (:obj:`~TesseractFeatureType`, optional):
                When set, aggregate the GCV output with respect to the
                specified aggregation level. Defaults to `None`.
        """

        res = self._detect(image)

        if return_response:
            return res

        if return_only_text:
            return res["text"]

        if agg_output_level is not None:
            return self.gather_data(res, agg_output_level)

        return res["text"]

    @staticmethod
    def gather_data(response, agg_level):
        """
        Gather the OCR'ed text, bounding boxes, and confidence
        in a given aggeragation level.
        """
        assert isinstance(
            agg_level, TesseractFeatureType
        ), f"Invalid agg_level {agg_level}"
        res = response["data"]
        df = (
            res[~res.text.isna()]
            .groupby(agg_level.group_levels)
            .apply(
                lambda gp: pd.Series(
                    [
                        gp["left"].min(),
                        gp["top"].min(),
                        gp["width"].max(),
                        gp["height"].max(),
                        gp["conf"].mean(),
                        gp["text"].str.cat(sep=" "),
                    ]
                )
            )
            .reset_index(drop=True)
            .reset_index()
            .rename(
                columns={
                    0: "x_1",
                    1: "y_1",
                    2: "w",
                    3: "h",
                    4: "score",
                    5: "text",
                    "index": "id",
                }
            )
            .assign(
                x_2=lambda x: x.x_1 + x.w,
                y_2=lambda x: x.y_1 + x.h,
                block_type="rectangle",
            )
            .drop(columns=["w", "h"])
        )

        return load_dataframe(df)

    @staticmethod
    def load_response(filename):
        with open(filename, "rb") as fp:
            res = pickle.load(fp)
        return res

    @staticmethod
    def save_response(res, file_name):

        with open(file_name, "wb") as fp:
            pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)
