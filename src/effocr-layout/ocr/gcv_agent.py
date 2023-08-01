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
import os
import json
import warnings

import numpy as np
from cv2 import imencode

from .base import BaseOCRAgent, BaseOCRElementType
from ..elements import Layout, TextBlock, Quadrilateral, TextBlock
from ..file_utils import is_gcv_available

if is_gcv_available():
    import google.protobuf.json_format as _json_format
    import google.cloud.vision as _vision



def _cvt_GCV_vertices_to_points(vertices):
    return np.array([[vertex.x, vertex.y] for vertex in vertices])


class GCVFeatureType(BaseOCRElementType):
    """
    The element types from Google Cloud Vision API
    """

    PAGE = 0
    BLOCK = 1
    PARA = 2
    WORD = 3
    SYMBOL = 4

    @property
    def attr_name(self):
        name_cvt = {
            GCVFeatureType.PAGE: "pages",
            GCVFeatureType.BLOCK: "blocks",
            GCVFeatureType.PARA: "paragraphs",
            GCVFeatureType.WORD: "words",
            GCVFeatureType.SYMBOL: "symbols",
        }
        return name_cvt[self]

    @property
    def child_level(self):
        child_cvt = {
            GCVFeatureType.PAGE: GCVFeatureType.BLOCK,
            GCVFeatureType.BLOCK: GCVFeatureType.PARA,
            GCVFeatureType.PARA: GCVFeatureType.WORD,
            GCVFeatureType.WORD: GCVFeatureType.SYMBOL,
            GCVFeatureType.SYMBOL: None,
        }
        return child_cvt[self]


class GCVAgent(BaseOCRAgent):
    """A wrapper for `Google Cloud Vision (GCV) <https://cloud.google.com/vision>`_ Text
    Detection APIs.

    Note:
        Google Cloud Vision API returns the output text in two types:

        * `text_annotations`:

            In this format, GCV automatically find the best aggregation
            level for the text, and return the results in a list. We use
            :obj:`~gather_text_annotations` to reterive this type of
            information.

        * `full_text_annotation`:

            To support better user control, GCV also provides the
            `full_text_annotation` output, where it returns the hierarchical
            structure of the output text. To process this output, we provide
            the :obj:`~gather_full_text_annotation` function to aggregate the
            texts of the given aggregation level.
    """

    DEPENDENCIES = ["google-cloud-vision"]
    
    def __init__(self, languages=None, ocr_image_decode_type=".png"):
        """Create a Google Cloud Vision OCR Agent.

        Args:
            languages (:obj:`list`, optional):
                You can specify the language code of the documents to detect to improve
                accuracy. The supported language and their code can be found on `this page
                <https://cloud.google.com/vision/docs/languages>`_.
                Defaults to None.

            ocr_image_decode_type (:obj:`str`, optional):
                The format to convert the input image to before sending for GCV OCR.
                Defaults to `".png"`.

                    * `".png"` is suggested as it does not compress the image.
                    * But `".jpg"` could also be a good choice if the input image is very large.
        """
        try:
            self._client = _vision.ImageAnnotatorClient()
        except:
            warnings.warn(
                "The GCV credential has not been set. You could not run the detect command."
            )
        self._context = _vision.types.ImageContext(language_hints=languages)
        self.ocr_image_decode_type = ocr_image_decode_type

    @classmethod
    def with_credential(cls, credential_path, **kwargs):
        """Specifiy the credential to use for the GCV OCR API.

        Args:
            credential_path (:obj:`str`): The path to the credential file
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
        return cls(**kwargs)

    def _detect(self, img_content):
        img_content = _vision.types.Image(content=img_content)
        response = self._client.document_text_detection(
            image=img_content, image_context=self._context
        )
        return response

    def detect(
        self,
        image,
        return_response=False,
        return_only_text=False,
        agg_output_level=None,
    ):
        """Send the input image for OCR.

        Args:
            image (:obj:`np.ndarray` or :obj:`str`):
                The input image array or the name of the image file
            return_response (:obj:`bool`, optional):
                Whether directly return the google cloud response.
                Defaults to `False`.
            return_only_text (:obj:`bool`, optional):
                Whether return only the texts in the OCR results.
                Defaults to `False`.
            agg_output_level (:obj:`~GCVFeatureType`, optional):
                When set, aggregate the GCV output with respect to the
                specified aggregation level. Defaults to `None`.
        """
        if isinstance(image, np.ndarray):
            img_content = imencode(self.ocr_image_decode_type, image)[1].tostring()

        elif isinstance(image, str):
            with io.open(image, "rb") as image_file:
                img_content = image_file.read()

        res = self._detect(img_content)

        if return_response:
            return res

        if return_only_text:
            return res.full_text_annotation.text

        if agg_output_level is not None:
            return self.gather_full_text_annotation(res, agg_output_level)

        return self.gather_text_annotations(res)

    @staticmethod
    def gather_text_annotations(response):
        """Convert the text_annotations from GCV output to an :obj:`Layout` object.

        Args:
            response (:obj:`AnnotateImageResponse`):
                The returned Google Cloud Vision AnnotateImageResponse object.

        Returns:
            :obj:`Layout`: The reterived layout from the response.
        """

        # The 0th element contains all texts
        doc = response.text_annotations[1:]
        gathered_text = Layout()

        for i, text_comp in enumerate(doc):
            points = _cvt_GCV_vertices_to_points(text_comp.bounding_poly.vertices)
            gathered_text.append(
                TextBlock(block=Quadrilateral(points), text=text_comp.description, id=i)
            )

        return gathered_text

    @staticmethod
    def gather_full_text_annotation(response, agg_level):
        """Convert the full_text_annotation from GCV output to an :obj:`Layout` object.

        Args:
            response (:obj:`AnnotateImageResponse`):
                The returned Google Cloud Vision AnnotateImageResponse object.

            agg_level (:obj:`~GCVFeatureType`):
                The layout level to aggregate the text in full_text_annotation.

        Returns:
            :obj:`Layout`: The reterived layout from the response.
        """

        def iter_level(
            iter,
            agg_level=None,
            text_blocks=None,
            texts=None,
            cur_level=GCVFeatureType.PAGE,
        ):

            for item in getattr(iter, cur_level.attr_name):
                if cur_level == agg_level:
                    texts = []

                # Go down levels to fetch the texts
                if cur_level == GCVFeatureType.SYMBOL:
                    texts.append(item.text)
                elif (
                    cur_level == GCVFeatureType.WORD
                    and agg_level != GCVFeatureType.SYMBOL
                ):
                    chars = []
                    iter_level(
                        item, agg_level, text_blocks, chars, cur_level.child_level
                    )
                    texts.append("".join(chars))
                else:
                    iter_level(
                        item, agg_level, text_blocks, texts, cur_level.child_level
                    )

                if cur_level == agg_level:
                    nonlocal element_id
                    points = _cvt_GCV_vertices_to_points(item.bounding_box.vertices)
                    text_block = TextBlock(
                        block=Quadrilateral(points),
                        text=" ".join(texts),
                        score=item.confidence,
                        id=element_id,
                    )

                    text_blocks.append(text_block)
                    element_id += 1

        if agg_level == GCVFeatureType.PAGE:
            doc = response.text_annotations[0]
            points = _cvt_GCV_vertices_to_points(doc.bounding_poly.vertices)

            text_blocks = [TextBlock(block=Quadrilateral(points), text=doc.description)]

        else:
            doc = response.full_text_annotation
            text_blocks = []
            element_id = 0
            iter_level(doc, agg_level, text_blocks)

        return Layout(text_blocks)

    def load_response(self, filename):
        with open(filename, "r") as f:
            data = f.read()
        return _json_format.Parse(
            data, _vision.types.AnnotateImageResponse(), ignore_unknown_fields=True
        )

    def save_response(self, res, file_name):
        res = _json_format.MessageToJson(res)

        with open(file_name, "w") as f:
            json_file = json.loads(res)
            json.dump(json_file, f)