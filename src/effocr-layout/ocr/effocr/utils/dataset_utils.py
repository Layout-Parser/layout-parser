import numpy as np
from torchvision import transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import time
from timeit import default_timer as timer


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class MedianPadWord:
    """This padding preserves the aspect ratio of the image. It also pads the image with the median value of the border pixels.
    Note how it also centres the ROI in the padded image."""
    def __init__(self, override=None,aspect_cutoff=0):
        self.override = override
        self.aspect_cutoff=aspect_cutoff
    def __call__(self, image):
        ##Convert to RGB
        image = image.convert("RGB") if isinstance(image, Image.Image) else image
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        max_side = max(image.size)
        aspect_ratio = image.size[0] / image.size[1]
        if aspect_ratio<self.aspect_cutoff:
            pad_x, pad_y = [int(0.75*max_side) for _ in image.size]
        else:
            pad_x, pad_y = [max_side - s for s in image.size]
        padding = (round((10+pad_x)/2), round((5+pad_y)/2), round((10+pad_x)/2), round((5+pad_y)/2)) ##Added some extra to avoid info on the long edge

        imgarray = np.array(image)
        h, w , c= imgarray.shape
        rightb, leftb = imgarray[:,w-1,:], imgarray[:,0,:]
        topb, bottomb = imgarray[0,:,:], imgarray[h-1,:,:]
        bordervals = np.concatenate([rightb, leftb, topb, bottomb], axis=0)
        medval = tuple([int(v) for v in np.median(bordervals, axis=0)])
        return T.Pad(padding, fill=medval if self.override is None else self.override)(image)
    
class MedianPad:

    def __init__(self, fill):
        self.fill = fill

    def __call__(self, imgarray):
        max_side = max(imgarray.shape)
        pad_y, pad_x, _ = [max_side - s for s in imgarray.shape]
        padding = (0, 0, pad_x, pad_y)
        pil_im = Image.fromarray(imgarray)
        return T.Pad(padding, fill=self.fill)(pil_im)


def timerhelper(s, x):
    print(s, timer())
    time.sleep(1)
    return x


def create_paired_transform(lang, size=224):
    return T.Compose([
        MedianPad(fill=(255,255,255)),
        T.ToTensor(),
        T.Resize((size, size)),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        lambda x: x.unsqueeze(0)
    ])

def create_paired_transform_word(lang, size=224,aspect_cutoff=0):
    return T.Compose([
        # SquarePad(),
        MedianPadWord(aspect_cutoff=aspect_cutoff),
        # T.Resize(size=(224,224)),
        # patch_resize,
        T.ToTensor(),
        T.Resize((size, size)),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        lambda x: x.unsqueeze(0)
        # featx_transform,
    ])
