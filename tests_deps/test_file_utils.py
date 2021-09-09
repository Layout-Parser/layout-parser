import pytest

from layoutparser import requires_backends

def test_when_backends_are_not_loaded():

    # When all the backeds are not installed, it should 
    # elicit only ImportErrors

    for backend_name in ["torch", "detectron2", "paddle", "effdet", "pytesseract", "google-cloud-vision"]:
        with pytest.raises(ImportError):
            requires_backends("a", backend_name)