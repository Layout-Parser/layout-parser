import os
import tarfile
from functools import reduce
from PIL import Image
import numpy as np

import requests
from tqdm import tqdm

from .preprocess import decode_image, resize, normalize_image, permute
from .catalog import PathManager, LABEL_MAP_CATALOG
from ..base_layoutmodel import BaseLayoutModel
from ...elements import Rectangle, TextBlock, Layout


__all__ = ["PaddleDetectionLayoutModel"]



class PaddleDetectionLayoutModel(BaseLayoutModel):
    """
    Args:
        config (object):
            config of model, defined by `Config(model_dir)`
        model_path (str):
            The path to the saved weights of the model.
        threshold (float):
            threshold to reserve the result for output
        input_shape(list):
            the image shape after reshape
        batch_size(int)：
            test batch size
        label_map (:obj:`dict`, optional):
            The map from the model prediction (ids) to realword labels (strings).
        enforce_cpu (bool):
            whether use cpu, if false, indicates use GPU
        enable_mkldnn(bool):
            whether use mkldnn to accelerate the computation
        thread_num(int):
            the number of threads
    Examples::
        >>> import layoutparser as lp
        >>> model = lp.models.PaddleDetectionLayoutModel('
                                    lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config')
        >>> model.detect(image)
    """
    DEPENDENCIES = ["paddlepaddle"]
    MODULES = [
        {
            "import_name": "_inference",
            "module_path": "paddle.inference",
        },
    ]
    DETECTOR_NAME = "paddledetection"

    def __init__(self,
                 config_path=None,
                 model_path=None,
                 label_map=None,
                 enforce_cpu=False,
                 extra_config={}):
        if config_path is not None and config_path.startswith("lp://"):
            prefix = "lp://"
            if label_map is None:
                dataset_name = config_path.lstrip("lp://").split("/")[0]
                label_map = LABEL_MAP_CATALOG[dataset_name]
            model_name = config_path[len(prefix) :].split('/')[1]
            config_path = self._reconstruct_path_with_detector_name(config_path)
            model_tar = PathManager.get_local_path(config_path)

            pre_dir = os.path.dirname(model_tar)
            base_dir = os.path.splitext(os.path.basename(model_tar))[0]
            model_dir = os.path.join(pre_dir, base_dir)
            self.untar_files(model_tar, model_dir)
        if model_path is not None:
            model_dir = model_path
        self.predictor = self.load_predictor(
            model_dir,
            batch_size=extra_config.get('batch_size',1),
            enforce_cpu=enforce_cpu,
            enable_mkldnn=extra_config.get('enable_mkldnn',True),
            thread_num=extra_config.get('thread_num',10))

        self.threshold = extra_config.get('threshold',0.5)
        self.input_shape = extra_config.get('input_shape',[3,640,640])
        self.label_map = label_map
        self.im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': None,
            'input_shape': self.input_shape,
        }

    def _reconstruct_path_with_detector_name(self, path: str) -> str:
        """This function will add the detector name (paddleDetection) into the
        lp model config path to get the "canonical" model name.

        For example,
        for a given config_path `lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config`,it will
        transform it into `lp://paddledetection/PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config`.
        However, if the config_path already contains the detector name, we won't change it.

        This function is a general step to support multiple backends in the layout-parser
        library.

        Args:
            path (str): The given input path that might or might not contain the detector name.

        Returns:
            str: a modified path that contains the detector name.
        """
        if path.startswith("lp://"):  # TODO: Move "lp://" to a constant
            model_name = path[len("lp://") :]
            model_name_segments = model_name.split("/")
            if (
                len(model_name_segments) == 3
                and "paddleDetection" not in model_name_segments
            ):
                return "lp://" + self.DETECTOR_NAME + "/" + path[len("lp://") :]
        return path

    def load_predictor(self,
                    model_dir,
                    batch_size=1,
                    enforce_cpu=False,
                    enable_mkldnn=True,
                    thread_num=10):
        """set AnalysisConfig, generate AnalysisPredictor
        Args:
            model_dir (str): root path of __model__ and __params__
            enforce_cpu (bool): whether use cpu
        Returns:
            predictor (PaddlePredictor): AnalysisPredictor
        Raises:
            ValueError: predict by TensorRT need enforce_cpu == False.
        """

        use_calib_mode = False
        config = self._inference.Config(
            os.path.join(model_dir, 'inference.pdmodel'),
            os.path.join(model_dir, 'inference.pdiparams'))

        if not enforce_cpu:
            # initial GPU memory(M), device ID
            # 2000 is an appropriate value for PaddleDetection model
            config.enable_use_gpu(2000, 0)
            # optimize graph and fuse op
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(thread_num)
            if enable_mkldnn:
                config.enable_mkldnn()

        # disable print log when predict
        config.disable_glog_info()
        # enable shared memory
        config.enable_memory_optim()
        # disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
        predictor = self._inference.create_predictor(config)
        return predictor

    def preprocess(self, image):
        """ preprocess image
        Args:
            image (np.ndarray): image (np.ndarray)
        Returns:
            inputs (dict): input of model
        """
        # read rgb image
        image, im_info = decode_image(image, self.im_info)
        # resize image by target_size and max_size
        image, im_info = resize(image, im_info)
        # normalize image
        image, im_info = normalize_image(image, im_info)
        # transpose images
        image, im_info = permute(image, im_info)

        inputs = {}
        inputs['image'] = np.array((image, )).astype('float32')
        inputs['im_shape'] = np.array((im_info['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info['scale_factor'], )).astype('float32')
        return inputs

    def gather_output(self, np_boxes, np_masks):
        """process output"""
        layout = Layout()
        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            results = {'boxes': np.array([])}
        else:
            results = {}
            results['boxes'] = np_boxes
            if np_masks is not None:
                results['masks'] = np_masks

        np_boxes = results['boxes']
        expect_boxes = (np_boxes[:, 1] > self.threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]

        for np_box in np_boxes:
            clsid, bbox, score = int(np_box[0]), np_box[2:], np_box[1]
            x_1, y_1, x_2, y_2 = bbox

            if self.label_map is not None:
                label = self.label_map[clsid]

            cur_block = TextBlock(
                Rectangle(x_1, y_1, x_2, y_2), type=label, score=score
            )
            layout.append(cur_block)

        return layout

    def detect(self,
                image):
        """Detect the layout of a given image.
        Args:
            image (:obj:`np.ndarray` or `PIL.Image`): The input image to detect.
        Returns:
            :obj:`~layoutparser.Layout`: The detected layout of the input image
        """
        # Convert PIL Image Input
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image)

        inputs = self.preprocess(image)

        np_boxes, np_masks = None, None
        input_names = self.predictor.get_input_names()
        for i,input_name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(input_name)
            input_tensor.copy_from_cpu(inputs[input_name])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()

        layout = self.gather_output(np_boxes, np_masks)
        return layout

    def untar_files(self, model_tar, model_dir):
        """ untar model files"""
        # including files after decompression
        tar_file_name_list = [
            'inference.pdiparams', 'inference.pdiparams.info', 'inference.pdmodel'
        ]
        if not os.path.exists(
                os.path.join(model_dir, 'inference.pdiparams')
        ) or not os.path.exists(
                os.path.join(model_dir, 'inference.pdmodel')):
            # the path to save the decompressed file
            os.makedirs(model_dir, exist_ok=True)
            with tarfile.open(model_tar, 'r') as tarobj:
                for member in tarobj.getmembers():
                    filename = None
                    for tar_file_name in tar_file_name_list:
                        if tar_file_name in member.name:
                            filename = tar_file_name
                    if filename is None:
                        continue
                    file = tarobj.extractfile(member)
                    with open(
                            os.path.join(model_dir, filename),
                            'wb') as model_file:
                        model_file.write(file.read())
        