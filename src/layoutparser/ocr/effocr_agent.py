import io
import os
import json
import warnings

import numpy as np
from cv2 import imencode
import multiprocessing
import faiss
from huggingface_hub import hf_hub_download
import joblib

from .base import BaseOCRAgent, BaseOCRElementType
from .effocr import EffLocalizer, EffRecognizer, EffLineDetector, \
                    run_effocr_word, create_paired_transform, create_paired_transform_word

EFFOCR_DEFAULT_CONFIG = {
    "line_model": "",
    "line_backend": "yolov8",
    "line_input_shape": (640, 640),
    "localizer_model": "",
    "localizer_backend": "yolov8",
    "localizer_input_shape": (640, 640),
    "word_recognizer_model": "./src/layoutparser/models/effocr/word_recognizer/enc.onnx",
    "word_index": "./src/layoutparser/models/effocr/word_recognizer/word_index.index",
    "word_ref": "./src/layoutparser/models/effocr/word_recognizer/word_ref.txt",
    "char_recognizer_model": "./src/layoutparser/models/effocr/char_recognizer/enc.onnx",
    "char_index": "./src/layoutparser/models/effocr/char_recognizer/char_index.index",
    "char_ref": "./src/layoutparser/models/effocr/char_recognizer/char_ref.txt",
    "localizer_iou_thresh": 0.10,
    "localizer_conf_thresh": 0.20,
    "line_iou_thresh": 0.05,
    "line_conf_thresh": 0.50,
    "word_dist_thresh": 0.90,
    "lang": "en",
}

HUGGINGFACE_MODEL_MAP = {
    'line_model': 'line.onnx',
    'localizer_model': 'localizer.onnx',
    'word_recognizer_model': 'word_recognizer/enc.onnx', 
    'char_recognizer_model': 'char_recognizer/enc.onnx',
    'word_index': 'word_recognizer/word_index.index',
    'word_ref': 'word_recognizer/word_ref.txt',
    'char_index': 'char_recognizer/char_index.index',
    'char_ref': 'char_recognizer/char_ref.txt'
}

HUGGINGFACE_REPO_NAME = 'dell-research-harvard/effocr_en'

class EffOCRFeatureType(BaseOCRElementType):
    """
    The element types from EffOCR
    """

    PAGE = 0
    PARA = 1
    LINE = 2
    WORD = 3
    CHAR = 4

    @property
    def attr_name(self):
        name_cvt = {
            EffOCRFeatureType.BLOCK: "blocks",
            EffOCRFeatureType.PARA: "paragraphs",
            EffOCRFeatureType.LINE: "lines",
            EffOCRFeatureType.WORD: "words",
        }
        return name_cvt[self]

    @property
    def child_level(self):
        child_cvt = {
            EffOCRFeatureType.BLOCK: EffOCRFeatureType.PARA,
            EffOCRFeatureType.PARA: EffOCRFeatureType.LINE,
            EffOCRFeatureType.LINE: EffOCRFeatureType.WORD,
            EffOCRFeatureType.WORD: None,
        }
        return child_cvt[self]



class EffOCRAgent(BaseOCRAgent):
    """EffOCR Inference -- Implements method described in https://scholar.harvard.edu/sites/scholar.harvard.edu/files/dell/files/effocr.pdf

    Note:
        TODO: Fill in with info once implemented
    """

    # TODO: Fill in with package dependencies
    DEPENDENCIES = ["effocr"]

    def __init__(self, languages="eng", **kwargs):
        """Create a EffOCR Agent.

        Args:
            languages (:obj:`list` or :obj:`str`, optional):
                You can specify the language code(s) of the documents to detect to determine the 
                language EffOCR uses when transcribing the document. As of 7/24, the only option is
                English, but Japanese EffOCR will be implemented soon. 
                Defaults to 'eng'.
        """
        if languages != 'eng':
            raise NotImplementedError("EffOCR only supports English at this time.")
        
        self.lang = languages if isinstance(languages, str) else "+".join(languages)
        
        self.config = EFFOCR_DEFAULT_CONFIG
        for key, value in kwargs.items():
            if key in self.config.keys():
                self.config[key] = value
            else:
                warnings.warn(f"Unknown config parameter {key} for {self.__class__.__name__}. Ignoring it.")

        self._check_and_download_models()
        self._check_and_download_indices()
        self._load_models()
        self._load_indices()
        print(self.config)

    def _check_and_download_models(self):
        '''
        Checks if all of line, localizer, word recognizer, and char recognizer are downloaded, 
        then downloads them if they are not.
        '''

        model_keys = ['line_model', 'localizer_model', 'word_recognizer_model', 'char_recognizer_model']
        for key in model_keys:  
            if not os.path.exists(self.config[key]) or not self.config[key].endswith('.onnx'):
                self.config[key] = hf_hub_download(HUGGINGFACE_REPO_NAME, HUGGINGFACE_MODEL_MAP[key])
                # TODO: replace FileNotFoundError with download code

    def _check_and_download_indices(self):
        '''
        Checks if the word and character recognizers' indices and refernece files are downloaded, 
        then downloads them if they are not.
        '''

        index_keys = ['word_index', 'char_index']
        ref_keys = ['word_ref', 'char_ref']

        for key in index_keys:
            if not os.path.exists(self.config[key]):
                self.config[key] = hf_hub_download(HUGGINGFACE_REPO_NAME, HUGGINGFACE_MODEL_MAP[key])
        
        for key in ref_keys:
            if not os.path.exists(self.config[key]):
                self.config[key] = hf_hub_download(HUGGINGFACE_REPO_NAME, HUGGINGFACE_MODEL_MAP[key])

    def _load_models(self):
        '''
        Function to instantiate each of the line model, 
        localizer model, word recognizer model, and char recognizer model.
        '''

        self.localizer_engine = EffLocalizer(
                self.config['localizer_model'],
                iou_thresh = self.config['localizer_iou_thresh'],
                conf_thresh = self.config['localizer_conf_thresh'],
                vertical = False if self.config['lang'] == "en" else True,
                num_cores = multiprocessing.cpu_count(),
                model_backend = self.config['localizer_backend'],
                input_shape = self.config['localizer_input_shape']
            )
        
        # TODO: Fix imports for paired_transforms
        char_transform = create_paired_transform(lang='en')
        word_transform = create_paired_transform_word(lang='en')

        self.word_recognizer_engine = EffRecognizer(
                model = self.config['word_recognizer_model'],
                transform = char_transform,
                num_cores=multiprocessing.cpu_count(),
            )
        
        self.char_recognizer_engine = EffRecognizer(
            model = self.config['char_recognizer_model'],
            transform = char_transform,
            num_cores=multiprocessing.cpu_count(),
        )

        self.line_detector_engine = EffLineDetector(
            self.config['line_model'],
            iou_thresh = self.config['line_iou_thresh'],
            conf_thresh = self.config['line_conf_thresh'],
            num_cores = multiprocessing.cpu_count(),
            model_backend = self.config['line_backend'],
            input_shape = self.config['line_input_shape']
        )

    def _load_indices(self):
        '''
        Function to instantiate the faiss indices for each of the word and character recognizers.
        Indicies are responsible for storing base vectors for each word/character and performing
        similarity search on unknown symbols.
        '''

        # char index
        self.char_index = faiss.read_index(self.config['char_index'])
        with open(self.config['char_ref']) as ref_file:
            self.candidate_chars = ref_file.read().split()

        # word index
        self.word_index = faiss.read_index(self.config['word_index'])
        with open(self.config['word_ref']) as ref_file:
            self.candidate_words = ref_file.read().split()

    def _detect(self, image, viz_lines_path=None):
        '''
        Function to detect text in an image using EffOCR.

        Each of the two main parts, line detection and line transcription, are abstrated out here
        '''

        # Line Detection
        line_crops, line_coords = self.line_detector_engine(image)

        # Line Transcription
        text_results = run_effocr_word(line_crops, self.localizer_engine, self.word_recognizer_engine, self.char_recognizer_engine, self.candidate_chars, 
                                       self.candidate_words, self.config['lang'], self.word_index, self.char_index, num_streams=multiprocessing.cpu_count(), vertical=False, 
                                       localizer_output = None, conf_thres=self.config['localizer_conf_thresh'], recognizer_thresh = self.config['word_dist_thresh'], 
                                       bbox_output = False, punc_padding = 0, insert_paragraph_breaks = True)
        
        return text_results

    def detect(self, image, return_response=False, return_only_text=True, agg_output_level=None, viz_lines_path = None):
        """Send the input image for OCR by the EffOCR agent.

        Args:
            image (:obj:`np.ndarray` or :obj:`str`):
                The input image array or the name of the image file
            return_response (:obj:`bool`, optional):
                Whether directly return the effocr output.
                Defaults to `False`.
            return_only_text (:obj:`bool`, optional):
                Whether return only the texts in the OCR results.
                Defaults to `False`.
            agg_output_level (:obj:`~EffOCRFeatureType`, optional):
                When set, aggregate the EffOCR output with respect to the
                specified aggregation level. Defaults to `None`.

        Returns:
            :obj:`dict` or :obj:`str`:
                The OCR results in the specified format.
        """

        res = self._detect(image, viz_lines_path = viz_lines_path)
        
        if return_response:
            return res

        if return_only_text:
            return res["text"]

        if agg_output_level is not None:
            return self.gather_data(res, agg_output_level)

        return res["text"]
    

if __name__ == '__main__':
    agent = EffOCRAgent()
    img_path = r'C:\Users\bryan\Documents\NBER\layout-parser\tests\fixtures\ocr\test_effocr_image.jpg'
