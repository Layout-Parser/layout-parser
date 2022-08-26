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

from layoutparser import (
    GCVAgent,
    GCVFeatureType,
    TesseractAgent,
    TesseractFeatureType,
    PaddleOCRAgent,
)
import json, cv2, os

image = cv2.imread("tests/fixtures/ocr/test_gcv_image.jpg")


def test_gcv_agent(test_detect=False):

    # Test loading the agent with designated credential
    ocr_agent = GCVAgent()

    # Test loading the saved response and parse the data
    res = ocr_agent.load_response("tests/fixtures/ocr/test_gcv_response.json")
    r0 = ocr_agent.gather_text_annotations(res)
    r1 = ocr_agent.gather_full_text_annotation(res, GCVFeatureType.SYMBOL)
    r2 = ocr_agent.gather_full_text_annotation(res, GCVFeatureType.WORD)
    r3 = ocr_agent.gather_full_text_annotation(res, GCVFeatureType.PARA)
    r4 = ocr_agent.gather_full_text_annotation(res, GCVFeatureType.BLOCK)
    r5 = ocr_agent.gather_full_text_annotation(res, GCVFeatureType.PAGE)

    # Test with a online image detection and compare the results with the stored one
    # Warning: there could be updates on the GCV side. So it would be good to not
    # frequently test this part.
    if test_detect:
        res2 = ocr_agent.detect(image, return_response=True)

        assert res == res2
        assert r0 == ocr_agent.gather_text_annotations(res2)
        assert r1 == ocr_agent.gather_full_text_annotation(res2, GCVFeatureType.SYMBOL)
        assert r2 == ocr_agent.gather_full_text_annotation(res2, GCVFeatureType.WORD)
        assert r3 == ocr_agent.gather_full_text_annotation(res2, GCVFeatureType.PARA)
        assert r4 == ocr_agent.gather_full_text_annotation(res2, GCVFeatureType.BLOCK)
        assert r5 == ocr_agent.gather_full_text_annotation(res2, GCVFeatureType.PAGE)

    # Finally, test the response storage and remove the file
    ocr_agent.save_response(res, "tests/fixtures/ocr/.test_gcv_response.json")
    os.remove("tests/fixtures/ocr/.test_gcv_response.json")


def test_tesseract(test_detect=False):

    ocr_agent = TesseractAgent(languages="eng")
    res = ocr_agent.load_response("tests/fixtures/ocr/test_tesseract_response.pickle")
    r0 = res["text"]
    r1 = ocr_agent.gather_data(res, agg_level=TesseractFeatureType.PAGE)
    r2 = ocr_agent.gather_data(res, agg_level=TesseractFeatureType.BLOCK)
    r3 = ocr_agent.gather_data(res, agg_level=TesseractFeatureType.PARA)
    r4 = ocr_agent.gather_data(res, agg_level=TesseractFeatureType.LINE)
    r5 = ocr_agent.gather_data(res, agg_level=TesseractFeatureType.WORD)

    # The results could be different is using another version of Tesseract Engine.
    # tesseract 4.1.1 is used for generating the pickle test file.
    if test_detect:
        res = ocr_agent.detect(image, return_response=True)
        assert r0 == res["text"]
        assert r1 == ocr_agent.gather_data(res, agg_level=TesseractFeatureType.PAGE)
        assert r2 == ocr_agent.gather_data(res, agg_level=TesseractFeatureType.BLOCK)
        assert r3 == ocr_agent.gather_data(res, agg_level=TesseractFeatureType.PARA)
        assert r4 == ocr_agent.gather_data(res, agg_level=TesseractFeatureType.LINE)
        assert r5 == ocr_agent.gather_data(res, agg_level=TesseractFeatureType.WORD)


def test_paddleocr(test_detect=False):

    ocr_agent = PaddleOCRAgent(languages="en")

    # The results could be different is using another version of PaddleOCR Engine.
    # PaddleOCR 2.0.1 is used for generating the result.
    if test_detect:
        res = ocr_agent.detect(image)
        print(res)