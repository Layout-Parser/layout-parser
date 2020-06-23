from layoutparser.models import * 
import cv2

def test_Detectron2Model():
    
    model = Detectron2LayoutModel('tests/source/config.yml')
    image = cv2.imread("tests/source/test_gcv_image.jpg")
    layout = model.detect(image)