# Model Zoo

We provide a spectrum of pre-trained models on different datasets.

## Example Usage: 

```python
import layoutparser as lp
model = lp.Detectron2LayoutModel(
            config='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', # In model catalog
            label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}, # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
        )
model.detect(image)
```

## Model Catalog

| Dataset                                                      | Model                                                        | Config Path                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------- |
| [HJDataset](https://dell-research-harvard.github.io/HJDataset/) | [faster_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/j4yseny2u0hn22r/config.yml?dl=1) | lp://HJDataset/faster_rcnn_R_50_FPN_3x/config |
| [HJDataset](https://dell-research-harvard.github.io/HJDataset/) | [mask_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/4jmr3xanmxmjcf8/config.yml?dl=1) | lp://HJDataset/mask_rcnn_R_50_FPN_3x/config   |
| [HJDataset](https://dell-research-harvard.github.io/HJDataset/) | [retinanet_R_50_FPN_3x](https://www.dropbox.com/s/z8a8ywozuyc5c2x/config.yml?dl=1) | lp://HJDataset/retinanet_R_50_FPN_3x/config   |
| [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)        | [faster_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/f3b12qc4hc0yh4m/config.yml?dl=1) | lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config |
| [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)        | [mask_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/u9wbsfwz4y0ziki/config.yml?dl=1) | lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config   |
| [PrimaLayout](https://www.primaresearch.org/dataset/)        | [mask_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/thdqhkvdihtr8yb/config.yml?dl=1) | lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config |

## Model `label_map`

| Dataset                                                      | Label Map                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [HJDataset](https://dell-research-harvard.github.io/HJDataset/) | `{1:"Page Frame", 2:"Row", 3:"Title Region", 4:"Text Region", 5:"Title", 6:"Subtitle", 7:"Other"}` |
| [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)        | `{1: "Text", 2: "Title", 3: "List", 4:"Table", 5:"Figure"}`     |
| [PrimaLayout](https://www.primaresearch.org/dataset/)        | `{1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}` |