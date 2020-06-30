from abc import ABC, abstractmethod
import os
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from ..elements import *
from fvcore.common.file_io import PathManager


__all__ = ['Detectron2LayoutModel']


class BaseLayoutModel(ABC):
    
    @abstractmethod
    def detect(self): pass


class Detectron2LayoutModel(BaseLayoutModel):
    """Create a Detectron2-based Layout Detection Model

    Args:
        config_path (:obj:`str`): 
            The path to the configuration file. 
        model_path (:obj:`str`, None): 
            The path to the saved weights of the model. 
            If set, overwrite the weights in the configuration file. 
            Defaults to `None`.
        label_map (:obj:`dict`, optional): 
            The map from the model prediction (ids) to real
            word labels (strings). 
            Defaults to `None`.
        extra_config (:obj:`list`, optional): 
            Extra configuration passed to the Detectron2 model 
            configuration. The argument will be used in the `merge_from_list 
            <https://detectron2.readthedocs.io/modules/config.html
            #detectron2.config.CfgNode.merge_from_list>`_ function. 
            Defaults to `[]`.
    
    Examples::
        >>> import layoutparser as lp
        >>> model = lp.models.Detectron2LayoutModel('lp://HJDataset/faster_rcnn_R_50_FPN_3x/config')
        >>> model.detect(image)
        
    """
    def __init__(self, config_path,
                       model_path = None,
                       label_map  = None,
                       extra_config= []):

        cfg = get_cfg()
        config_path = PathManager.get_local_path(config_path)
        cfg.merge_from_file(config_path)
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
                label = self.label_map.get(label, label)

            cur_block = TextBlock(
                    Rectangle(x_1, y_1, x_2, y_2),
                    type=label, 
                    score=score)
            layout.append(cur_block)

        return layout

    def _create_model(self):
        self.model = DefaultPredictor(self.cfg)

    def detect(self, image):
        """Detect the layout of a given image.

        Args:
            image (:obj:`np.ndarray`): The input image to detect.

        Returns:
            :obj:`~layoutparser.Layout`: The detected layout of the input image
        """
        outputs = self.model(image)
        layout  = self.gather_output(outputs)
        return layout