# Model Zoo

We provide a spectrum of pre-trained models on different datasets.

## Example Usage: 

```python
import layoutparser as lp
model = lp.Detectron2LayoutModel(
            config_path ='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', # In model catalog
            label_map   ={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}, # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
        )
model.detect(image)
```

## Model Catalog

| Dataset                                                               | Model                                                                                      | Config Path                                            | Eval Result (mAP)                                                         |
|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------------|
| [HJDataset](https://dell-research-harvard.github.io/HJDataset/)       | [faster_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/j4yseny2u0hn22r/config.yml?dl=1)       | lp://HJDataset/faster_rcnn_R_50_FPN_3x/config          |                                                                           |
| [HJDataset](https://dell-research-harvard.github.io/HJDataset/)       | [mask_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/4jmr3xanmxmjcf8/config.yml?dl=1)         | lp://HJDataset/mask_rcnn_R_50_FPN_3x/config            |                                                                           |
| [HJDataset](https://dell-research-harvard.github.io/HJDataset/)       | [retinanet_R_50_FPN_3x](https://www.dropbox.com/s/z8a8ywozuyc5c2x/config.yml?dl=1)         | lp://HJDataset/retinanet_R_50_FPN_3x/config            |                                                                           |
| [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)                 | [faster_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/f3b12qc4hc0yh4m/config.yml?dl=1)       | lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config          |                                                                           |
| [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)                 | [mask_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/u9wbsfwz4y0ziki/config.yml?dl=1)         | lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config            |                                                                           |
| [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)                 | [mask_rcnn_X_101_32x8d_FPN_3x](https://www.dropbox.com/s/nau5ut6zgthunil/config.yaml?dl=1) | lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config     | 88.98 [eval.csv](https://www.dropbox.com/s/15ytg3fzmc6l59x/eval.csv?dl=0) |
| [PrimaLayout](https://www.primaresearch.org/dataset/)                 | [mask_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/yc92x97k50abynt/config.yaml?dl=1)        | lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config          | 69.35 [eval.csv](https://www.dropbox.com/s/9uuql57uedvb9mo/eval.csv?dl=0) |
| [NewspaperNavigator](https://news-navigator.labs.loc.gov/)            | [faster_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/wnido8pk4oubyzr/config.yml?dl=1)       | lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config |                                                                           |
| [TableBank](https://doc-analysis.github.io/tablebank-page/index.html) | [faster_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/7cqle02do7ah7k4/config.yaml?dl=1)      | lp://TableBank/faster_rcnn_R_50_FPN_3x/config          | 89.78 [eval.csv](https://www.dropbox.com/s/1uwnz58hxf96iw2/eval.csv?dl=0) |
| [TableBank](https://doc-analysis.github.io/tablebank-page/index.html) | [faster_rcnn_R_101_FPN_3x](https://www.dropbox.com/s/h63n6nv51kfl923/config.yaml?dl=1)     | lp://TableBank/faster_rcnn_R_101_FPN_3x/config         | 91.26 [eval.csv](https://www.dropbox.com/s/e1kq8thkj2id1li/eval.csv?dl=0) |
| [Math Formula Detection(MFD)](http://transcriptorium.eu/~htrcontest/MathsICDAR2021/) | [faster_rcnn_R_50_FPN_3x](https://www.dropbox.com/s/ld9izb95f19369w/config.yaml?dl=1)     | lp://MFD/faster_rcnn_R_50_FPN_3x/config         | 79.68 [eval.csv](https://www.dropbox.com/s/1yvrs29jjybrlpw/eval.csv?dl=0) |


* For PubLayNet models, we suggest using `mask_rcnn_X_101_32x8d_FPN_3x` model as it's trained on the whole training set, while others are only trained on the validation set (the size is only around 1/50). You could expect a 15% AP improvement using the `mask_rcnn_X_101_32x8d_FPN_3x` model.

## Model `label_map`

| Dataset                                                      | Label Map                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [HJDataset](https://dell-research-harvard.github.io/HJDataset/) | `{1:"Page Frame", 2:"Row", 3:"Title Region", 4:"Text Region", 5:"Title", 6:"Subtitle", 7:"Other"}` |
| [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)        | `{0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}`     |
| [PrimaLayout](https://www.primaresearch.org/dataset/)        | `{1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}` |
| [NewspaperNavigator](https://news-navigator.labs.loc.gov/)        | `{0: "Photograph", 1: "Illustration", 2: "Map", 3: "Comics/Cartoon", 4: "Editorial Cartoon", 5: "Headline", 6: "Advertisement"}` |
| [TableBank](https://doc-analysis.github.io/tablebank-page/index.html)         | `{0: "Table"}` |
| [MFD](http://transcriptorium.eu/~htrcontest/MathsICDAR2021/)         | `{1: "Equation"}` |