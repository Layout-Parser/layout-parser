import os
import tarfile
from functools import reduce
from PIL import Image
import numpy as np

import requests
from tqdm import tqdm

from .preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
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
        enforce_mkldnn(bool):
            whether use mkldnn to accelerate the computation
        thread_num(int):
            the number of threads
        use_dynamic_shape (bool):
            use dynamic shape or not
        trt_min_shape (int):
            min shape for dynamic shape in trt
        trt_max_shape (int):
            max shape for dynamic shape in trt
        trt_opt_shape (int):
            opt shape for dynamic shape in trt.
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
    DETECTOR_NAME = "paddleDetection"

    def __init__(self,
                 config_path=None,
                 model_path=None,
                 threshold=0.5,
                 input_shape=[3,640,640],
                 batch_size=1,
                 label_map=None,
                 enforce_cpu=False,
                 enable_mkldnn=True,
                 thread_num=10,
                 use_dynamic_shape=False,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 min_subgraph_size=3):
        if config_path is not None and config_path.startswith("lp://"):
            prefix = "lp://"
            if label_map is None:
                dataset_name = config_path.lstrip("lp://").split("/")[0]
                label_map = LABEL_MAP_CATALOG[dataset_name]
            model_name = config_path[len(prefix) :].split('/')[1]
            config_path = self._reconstruct_path_with_detector_name(config_path)
            url = PathManager.get_local_path(config_path)
            base_dir = os.path.expanduser("~/.paddledet/")
            base_inference_model_dir = os.path.join(base_dir, 'inference_model')

            model_dir = os.path.join(base_inference_model_dir, model_name, model_name+'_infer')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            maybe_download(model_storage_directory=model_dir, url=url)
        if model_path is not None:
            model_dir = model_path
        self.predictor = self.load_predictor(
            model_dir,
            batch_size=batch_size,
            enforce_cpu=enforce_cpu,
            enable_mkldnn=enable_mkldnn,
            thread_num=thread_num,
            min_subgraph_size=min_subgraph_size,
            use_dynamic_shape=use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape)

        self.threshold = threshold
        self.input_shape = input_shape
        self.label_map = label_map

    def _reconstruct_path_with_detector_name(self, path: str) -> str:
        """This function will add the detector name (paddleDetection) into the
        lp model config path to get the "canonical" model name.

        For example,
        for a given config_path `lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config`,it will
        transform it into `lp://paddleDetection/PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config`.
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
                    thread_num=10,
                    min_subgraph_size=3,
                    use_dynamic_shape=False,
                    trt_min_shape=1,
                    trt_max_shape=1280,
                    trt_opt_shape=640):
        """set AnalysisConfig, generate AnalysisPredictor
        Args:
            model_dir (str): root path of __model__ and __params__
            enforce_cpu (bool): whether use cpu
            use_dynamic_shape (bool): use dynamic shape or not
            trt_min_shape (int): min shape for dynamic shape in trt
            trt_max_shape (int): max shape for dynamic shape in trt
            trt_opt_shape (int): opt shape for dynamic shape in trt
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

    def create_inputs(self, image, im_info):
        """generate input for different model type
        Args:
            image (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            inputs (dict): input of model
        """
        inputs = {}
        inputs['image'] = np.array((image, )).astype('float32')
        inputs['im_shape'] = np.array((im_info['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info['scale_factor'], )).astype('float32')

        return inputs

    def preprocess(self, image):
        """ preprocess image"""
        image, im_info = preprocess(image, self.input_shape)
        inputs = self.create_inputs(image, im_info)
        return inputs

    def postprocess(self, np_boxes, np_masks):
        """ postprocess output of predictor"""
        results = {}
        results['boxes'] = np_boxes
        if np_masks is not None:
            results['masks'] = np_masks
        return results

    def gather_output(self, results):
        """process output"""
        layout = Layout()
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
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
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

        # do not perform postprocess in benchmark mode
        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            results = {'boxes': np.array([])}
        else:
            results = self.postprocess(
                np_boxes, np_masks)

        layout = self.gather_output(results)
        return layout


def download_with_progressbar(url, save_path):
    """download model"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes:
        raise Exception(
            "Something went wrong while downloading model/image from {}".
            format(url))

def maybe_download(model_storage_directory, url):
    """ using custom model """
    tar_file_name_list = [
        'inference.pdiparams', 'inference.pdiparams.info', 'inference.pdmodel'
    ]
    if not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdiparams')
    ) or not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdmodel')):
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        print('download {} to {}'.format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path)
        with tarfile.open(tmp_path, 'r') as tarobj:
            for member in tarobj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if tar_file_name in member.name:
                        filename = tar_file_name
                if filename is None:
                    continue
                file = tarobj.extractfile(member)
                with open(
                        os.path.join(model_storage_directory, filename),
                        'wb') as file:
                    file.write(file.read())
        os.remove(tmp_path)
        