import os
import sys
# import mmcv
import torch
import numpy as np
import onnxruntime as ort
import torchvision
from torchvision.ops import nms
import cv2
import onnx
from math import floor, ceil

from .ops import non_max_suppression as yolov8_nms
from .ops import get_onnx_input_name
from ..utils import letterbox, non_max_suppression

DEFAULT_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DEFAULT_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

class EffLineDetector:
    """
    Class for running the EffOCR line detection model. Essentially a wrapper for the onnxruntime 
    inference session based on the model, wit some additional postprocessing, especially regarding splitting and 
    recombining especailly tall layout regions
    """

    def __init__(self, model_path, iou_thresh = 0.15, conf_thresh = 0.20, 
                    num_cores = None, providers=None, input_shape = (640, 640), model_backend='yolo',
                    min_seg_ratio = 2, visualize = None):
        """Instantiates the object, including setting up the wrapped ONNX InferenceSession

        Args:
            model_path (str): Path to ONNX model that will be used
            iou_thresh (float, optional): IOU filter for line detection NMS. Defaults to 0.15.
            conf_thresh (float, optional): Confidence filter for line detection NMS. Defaults to 0.20.
            num_cores (_type_, optional): Number of cores to use during inference. Defaults to None, meaning no intra op thread limit.
            providers (_type_, optional): Any particular ONNX providers to use. Defaults to None, meaning results of ort.get_available_providers() will be used.
            input_shape (tuple, optional): Shape of input images. Defaults to (640, 640).
            model_backend (str, optional): Original model backend being used. Defaults to 'yolo'. Options are mmdetection, detectron2, yolo, yolov8.
        """


        # Set up and instantiate a ort InfernceSession
        sess_options = ort.SessionOptions()
        if num_cores is not None:
            sess_options.intra_op_num_threads = num_cores

        if providers is None:
            providers = ort.get_available_providers()

        self._eng_net = ort.InferenceSession(
                    model_path,
                    sess_options,
                    providers=providers,
                )

        # Load in the model as a standard ONNX model to get the input shape and name
        base_model = onnx.load(model_path)
        self._input_name = get_onnx_input_name(base_model)
        self._model_input_shape = self._eng_net.get_inputs()[0].shape
        
        # Rest of the params
        self._iou_thresh = iou_thresh
        self._conf_thresh = conf_thresh

        if isinstance(self._model_input_shape[-1], int) and isinstance(self._model_input_shape[-2], int):
            self._input_shape = (self._model_input_shape[-2], self._model_input_shape[-1])
        else:
            self._input_shape = input_shape
        self._model_backend = model_backend
        self.min_seg_ratio = min_seg_ratio # Ratio that determines at what point the model will split a region into two



    def __call__(self, imgs, visualize = None):
        """Wraps the run method, allowing the object to be called directly

        Args:
            imgs (list or str or np.ndarray): List of image paths, list of images as np.ndarrays, or single image path, or single image as np.ndarray

        Returns:
            _type_: _description_
        """
        return self.run(imgs, visualize = visualize)
    
    def run(self, imgs, visualize = None):
        orig_img = imgs.copy()
        if isinstance(imgs, list):
            if all(isinstance(img, str) for img in imgs):
                imgs = [self.load_line_img(img, self._input_shape, backend=self._model_backend) for img in imgs]
            elif all(isinstance(img, np.ndarray) for img in imgs):
                imgs = [self.get_crops_from_layout_image(img) for img in imgs]
                imgs = [self.format_line_img(img, self._input_shape, backend=self._model_backend) for img in imgs]
            else:
                raise ValueError('Invalid combination if input types in Line Detection list! Must be all str or all np.ndarray')
        elif isinstance(imgs, str):
            imgs = [self.load_line_img(imgs, self._input_shape, backend=self._model_backend)]
        elif isinstance(imgs, np.ndarray):
            imgs = self.get_crops_from_layout_image(imgs)
            orig_shapes = [img.shape for img in imgs]
            imgs = [self.format_line_img(img, self._input_shape, backend=self._model_backend) for img in imgs]
        else:
            raise ValueError('Input type {} is not implemented'.format(type(imgs)))
        
        results = [self._eng_net.run(None, {self._input_name: img}) for img in imgs]
        return self._postprocess(results, imgs, orig_shapes, orig_img, viz_lines = visualize)

    def _postprocess(self, results, imgs, orig_shapes, orig_img, viz_lines = None):
        #YOLO NMS is carried out now, other backends will filter by bbox confidence score later
        if self._model_backend == 'yolo':  
            preds = [torch.from_numpy(pred[0]) for pred in results]
            preds = [non_max_suppression(pred, conf_thres = self._conf_thresh, iou_thres=self._iou_thresh, max_det=100)[0] for pred in preds]

        elif self._model_backend == 'yolov8':
            preds = [torch.from_numpy(pred[0]) for pred in results]
            preds = [yolov8_nms(pred, conf_thres = self._conf_thresh, iou_thres=self._iou_thresh, max_det=100)[0] for pred in preds]

        elif self._model_backend == 'detectron2' or self._model_backend == 'mmdetection':
            return results
    
        preds = self.adjust_line_preds(preds, imgs, orig_shapes)
        final_preds = self.readjust_line_predictions(preds, imgs[0].shape[1])

        line_crops, line_coords = [], []
        for i, line_proj_crop in enumerate(final_preds):
            x0, y0, x1, y1 = map(round, line_proj_crop)
            line_crop = orig_img[y0:y1, x0:x1]
            if line_crop.shape[0] == 0 or line_crop.shape[1] == 0:
                continue

            # Line crops becomes a list of tuples (bbox_id, line_crop [the image itself], line_proj_crop [the coordinates of the line in the layout image])
            line_crops.append(np.array(line_crop).astype(np.float32))
            line_coords.append((y0, x0, y1, x1))

            # If asked to visualize the line detections, draw a rectangle representing each line crop on the original image
            if viz_lines is not None:
                cv2.rectangle(orig_img, (x0, y0), (x1, y1), (255, 0, 0), 2)
                
        # If asked to visualize, output the image with the line detections drawn on it
        if viz_lines is not None:
            cv2.imwrite(viz_lines, orig_img)
            
        return line_crops, line_coords


    def adjust_line_preds(self, preds, imgs, orig_shapes):
        adjusted_preds = []

        for pred, shape in zip(preds, orig_shapes):
            line_predictions = pred[pred[:, 1].sort()[1]]
            line_bboxes, line_confs, line_labels = line_predictions[:, :4], line_predictions[:, -2], line_predictions[:, -1]

            im_width, im_height = shape[1], shape[0]
            if im_width > im_height:
                h_ratio = (im_height / im_width) * 640
                h_trans = 640 * ((1 - (im_height / im_width)) / 2)
            else:
                h_trans = 0
                h_ratio = 640

            line_proj_crops = []
            for line_bbox in line_bboxes:
                x0, y0, x1, y1 = torch.round(line_bbox)
                x0, y0, x1, y1 = 0, int(floor((y0.item() - h_trans) * im_height / h_ratio)), \
                                im_width, int(ceil((y1.item() - h_trans) * im_height  / h_ratio))
            
                line_proj_crops.append((x0, y0, x1, y1))
            
            adjusted_preds.append((line_proj_crops, line_confs, line_labels))

        return adjusted_preds
            
    def readjust_line_predictions(self, line_preds, orig_img_width):
        y0 = 0
        dif = int(orig_img_width * 1.5)
        all_preds, final_preds = [], []
        for j in range(len(line_preds)):
            preds, probs, labels = line_preds[j]
            for i, pred in enumerate(preds):
                all_preds.append((pred[0], pred[1] + y0, pred[2], pred[3] + y0, probs[i]))
            y0 += dif
        
        all_preds = torch.tensor(all_preds)
        if all_preds.dim() > 1:
            keep_preds = nms(all_preds[:, :4], all_preds[:, -1], iou_threshold=0.15)
            filtered_preds = all_preds[keep_preds, :4]
            filtered_preds = filtered_preds[filtered_preds[:, 1].sort()[1]]
            for pred in filtered_preds:
                x0, y0, x1, y1 = torch.round(pred)
                x0, y0, x1, y1 = x0.item(), y0.item(), x1.item(), y1.item()
                final_preds.append((x0, y0, x1, y1))
            return final_preds
        else:
            return []
             
    def format_line_img(self, img, input_shape, backend='yolo'):
        if backend == 'yolo' or backend == 'yolov8':
            im = letterbox(img, input_shape, stride=32, auto=False)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = im.astype(np.float32) / 255.0  # 0 - 255 to 0.0 - 1.0
            if im.ndim == 3:
                im = np.expand_dims(im, 0)

        elif backend == 'detectron2':
            im = letterbox(img, input_shape, stride=32, auto=False)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = im.astype(np.float32)
            
        elif backend == 'mmdetection':
            im = mmcv.imrescale(img, (input_shape[0], input_shape[1]))
            im = mmcv.impad(im, shape = input_shape, pad_val=0)
            im = mmcv.imnormalize(im, DEFAULT_MEAN, DEFAULT_STD, to_rgb=True)
            im = im.transpose(2, 0, 1)
            if im.ndim == 3:
                im = np.expand_dims(im, 0)

            
        else:
            raise NotImplementedError('Backend {} is not implemented'.format(backend))
        
        return im

    def load_line_img(self, input_path, input_shape, backend='yolo'):
        if backend == 'yolo' or backend == 'yolov8' or backend == 'detectron2':
            im0 = cv2.imread(input_path)
            im0 = self.get_crops_from_layout_image(im0)
            return [self.format_line_img(im, input_shape, backend=backend) for im in im0]
        elif backend == 'mmdetection':
            one_img = mmcv.imread(input_path)
            one_img = self.get_crops_from_layout_image(one_img)
            return [self.format_line_img(one_im, input_shape, backend=backend) for one_im in one_img]
        else:
            raise NotImplementedError('Backend {} is not implemented'.format(backend))

    def get_crops_from_layout_image(self, image):
        im_width, im_height = image.shape[0], image.shape[1]
        if im_height <= im_width * self.min_seg_ratio:
            return [image]
        else:
            y0 = 0
            y1 = im_width * self.min_seg_ratio
            crops = []
            while y1 <= im_height:
                crops.append(image.crop((0, y0, im_width, y1)))
                y0 += int(im_width * self.min_seg_ratio * 0.75) # .75 factor ensures there is overlap between crops
                y1 += int(im_width * self.min_seg_ration * 0.75)
            
            crops.append(image.crop((0, y0, im_width, im_height)))
            return crops