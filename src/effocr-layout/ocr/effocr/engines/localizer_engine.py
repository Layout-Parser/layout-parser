import os
import sys
# import mmcv
import torch
import numpy as np
import onnxruntime as ort
import torchvision
import cv2
import onnx

from .ops import non_max_suppression as yolov8_nms

DEFAULT_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DEFAULT_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

class EffLocalizer:

    def __init__(self, model_path, iou_thresh = 0.01, conf_thresh = 0.30, vertical = False, 
                    num_cores = None, providers=None, input_shape = (640, 640), model_backend='yolo'):
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

        base_model = onnx.load(model_path)
        self._input_name = EffLocalizer.get_onnx_input_name(base_model)
        self._model_input_shape = self._eng_net.get_inputs()[0].shape
        self._iou_thresh = iou_thresh
        self._conf_thresh = conf_thresh
        self._vertical = vertical

        if isinstance(self._model_input_shape[-1], int) and isinstance(self._model_input_shape[-2], int):
            self._input_shape = (self._model_input_shape[-2], self._model_input_shape[-1])
        else:
            self._input_shape = input_shape
        self._model_backend = model_backend



    def __call__(self, imgs):
        return self.run(imgs)
    
    def run(self, imgs):
        if isinstance(imgs, list):
            if isinstance(imgs[0], str):
                imgs = [EffLocalizer.load_localizer_img(img, self._input_shape, backend=self._model_backend) for img in imgs]
            else:
                imgs = [EffLocalizer.format_localizer_img(img, self._input_shape, backend=self._model_backend) for img in imgs]
        elif isinstance(imgs, str):
            imgs = [EffLocalizer.load_localizer_img(imgs, self._input_shape, backend=self._model_backend)] 
        elif isinstance(imgs, np.ndarray):
            imgs = [EffLocalizer.format_localizer_img(imgs, self._input_shape, backend=self._model_backend)]
        else:
            raise NotImplementedError('Input type {} is not implemented'.format(type(imgs)))
        
        results = [self._eng_net.run(None, {self._input_name: img}) for img in imgs]
        return self._postprocess(results)

    def _postprocess(self, results):
        #YOLO NMS is carried out now, other backends will filter by bbox confidence score later
        if self._model_backend == 'yolo':
            
            preds = [torch.from_numpy(pred[0]) for pred in results]
            preds = [self.non_max_suppression(pred, conf_thres = self._conf_thresh, iou_thres=self._iou_thresh, max_det=1000)[0] for pred in preds]
            return preds
        
        elif self._model_backend == 'yolov8':
            preds = [torch.from_numpy(pred[0]) for pred in results]
            preds = [yolov8_nms(pred, conf_thres = self._conf_thresh, iou_thres=self._iou_thresh, max_det=50)[0] for pred in preds]
            return preds

        elif self._model_backend == 'detectron2' or self._model_backend == 'mmdetection':
            return results
    
    @staticmethod
    def get_onnx_input_name(model):
        input_all = [node.name for node in model.graph.input]
        input_initializer =  [node.name for node in model.graph.initializer]
        net_feed_input = list(set(input_all)  - set(input_initializer))
        return net_feed_input[0]
    
    @staticmethod
    def format_localizer_img(img, input_shape, backend='yolo'):
        if backend == 'yolo' or backend == 'yolov8':
            im = EffLocalizer.letterbox(img, input_shape, stride=32, auto=False)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = im.astype(np.float32) / 255.0  # 0 - 255 to 0.0 - 1.0
            if im.ndim == 3:
                im = np.expand_dims(im, 0)
            return im
        elif backend == 'detectron2':
            im = EffLocalizer.letterbox(img, input_shape, stride=32, auto=False)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = im.astype(np.float32)
            return im
        elif backend == 'mmdetection':
            one_img = mmcv.imrescale(img, (input_shape[0], input_shape[1]))
            one_img = mmcv.impad(one_img, shape = input_shape, pad_val=0)
            one_img = mmcv.imnormalize(one_img, DEFAULT_MEAN, DEFAULT_STD, to_rgb=True)
            one_img = one_img.transpose(2, 0, 1)
            if one_img.ndim == 3:
                one_img = np.expand_dims(one_img, 0)

            return one_img
        else:
            raise NotImplementedError('Backend {} is not implemented'.format(backend))

    @staticmethod
    def load_localizer_img(input_path, input_shape, backend='yolo'):
        if backend == 'yolo' or backend == 'yolov8':
            im0 = cv2.imread(input_path)
            im = EffLocalizer.letterbox(im0, input_shape, stride=32, auto=False)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = im.astype(np.float32) / 255.0  # 0 - 255 to 0.0 - 1.0
            if im.ndim == 3:
                im = np.expand_dims(im, 0)
            return im
        elif backend == 'detectron2':
            im0 = cv2.imread(input_path)
            im = EffLocalizer.letterbox(im0, input_shape, stride=32, auto=False)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = im.astype(np.float32)
            return im
        elif backend == 'mmdetection':
            one_img = mmcv.imread(input_path)
            one_img = mmcv.imrescale(one_img, (input_shape[0], input_shape[1]))
            one_img = mmcv.impad(one_img, shape = input_shape, pad_val=0)
            one_img = mmcv.imnormalize(one_img, DEFAULT_MEAN, DEFAULT_STD, to_rgb=True)
            one_img = one_img.transpose(2, 0, 1)
            if one_img.ndim == 3:
                one_img = np.expand_dims(one_img, 0)

            return one_img
        else:
            raise NotImplementedError('Backend {} is not implemented'.format(backend))


    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    @staticmethod
    def box_iou(box1, box2, eps=1e-7):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    @staticmethod
    def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  ):

        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = EffLocalizer.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            else:
                x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = EffLocalizer.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)

        return output