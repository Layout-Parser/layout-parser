from layoutparser.ocr import GCVAgent, GCVFeatureType, TesseractAgent, TesseractFeatureType
import json, cv2, os

image = cv2.imread("tests/source/test_gcv_image.jpg")

def test_gcv_agent(test_detect=False):
    
    # Test loading the agent with designated credential
    ocr_agent = GCVAgent.with_credential("tests/source/test_gcv_credential.json", languages = ['en'])
    
    # Test loading the saved response and parse the data 
    res = ocr_agent.load_response("tests/source/test_gcv_response.json")
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
    ocr_agent.save_response(res, "tests/source/.test_gcv_response.json")
    os.remove("tests/source/.test_gcv_response.json")
   
def test_tesseract():

    ocr_agent = TesseractAgent(languages='eng')
    res = ocr_agent.detect(image, return_response=True)
    ocr_agent.gather_data(res, agg_level=TesseractFeatureType.PAGE)
    ocr_agent.gather_data(res, agg_level=TesseractFeatureType.BLOCK)
    ocr_agent.gather_data(res, agg_level=TesseractFeatureType.PARA)
    ocr_agent.gather_data(res, agg_level=TesseractFeatureType.LINE)
    ocr_agent.gather_data(res, agg_level=TesseractFeatureType.WORD)