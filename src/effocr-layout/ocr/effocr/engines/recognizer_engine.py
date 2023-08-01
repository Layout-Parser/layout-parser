import os
import sys
import torch
import onnxruntime as ort
import numpy as np


class EffRecognizer:

    def __init__(self, model, transform = None, num_cores = None, providers=None, char=True):
        
        sess_options = ort.SessionOptions()
        if num_cores is not None:
            sess_options.intra_op_num_threads = num_cores

        if providers is None:
            providers = ort.get_available_providers()

        self.transform = transform
        # null_input = torch.zeros((3, 224, 224)) if char else torch.zeros((1, 224, 224))
        self._eng_net = ort.InferenceSession(
                    model,
                    sess_options,
                    providers=providers,
                )
        
    def __call__(self, imgs):
        return self.run(imgs)
    
    def run(self, imgs):
        trans_imgs = []
        for img in imgs:
            try:
                trans_imgs.append(self.transform(img.astype(np.uint8))[0])
            except Exception as e:
                trans_imgs.append(torch.zeros((3, 224, 224)))

        onnx_input = torch.nn.functional.pad(torch.stack(trans_imgs), (0, 0, 0, 0, 0, 0, 0, 64 - len(imgs))).numpy()
        
        return self._eng_net.run(None, {'imgs': onnx_input})

