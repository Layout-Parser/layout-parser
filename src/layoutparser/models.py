from abc import ABC, abstractmethod
import os
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from .elements import *

__all__ = ['Detectron2LayoutModel']


class BaseLayoutModel(ABC):
    
    @abstractmethod
    def detect(self): pass


class Detectron2LayoutModel(BaseLayoutModel):

    def __init__(self, config_name,
                       model_path = None,
                       label_map  = None,
                       extra_config= []):

        cfg = get_cfg()
        cfg.merge_from_file(config_name)
        cfg.merge_from_list(extra_config)
        
        if model_path is not None:
            cfg.MODEL.WEIGHTS = model_path            
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = cfg
        
        self.label_map = label_map
        self._create_model()

    def gather_output(self, outputs):

        instance_pred = outputs['instances'].to("cpu")

        layout = Layout()
        scores = instance_pred.scores.tolist()
        boxes  = instance_pred.pred_boxes.tensor.tolist()
        labels = instance_pred.pred_classes.tolist()

        for score, box, label in zip(scores, boxes, labels):
            x_1, y_1, x_2, y_2 = box

            if self.label_map is not None:
                label = self.label_map[label]

            cur_block = TextBlock(
                    Rectangle(x_1, y_1, x_2, y_2),
                    type=label, 
                    score=score)
            layout.append(cur_block)

        return layout

    def _create_model(self):
        self.model = DefaultPredictor(self.cfg)

    def detect(self, image):

        outputs = self.model(image)
        layout  = self.gather_output(outputs)
        return layout