import layoutparser as lp
import gradio as gr

model = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config')

def lpi(img):
  layout = model.detect(img) # You need to load the image somewhere else, e.g., image = cv2.imread(...)
  return lp.draw_box(img, layout,) # With extra configurations

inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil",label="Output Image")

title = "Layout Parser"
description = "demo for Layout Parser. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2103.15348'>LayoutParser: A Unified Toolkit for Deep Learning Based Document Image Analysis</a> | <a href='https://github.com/Layout-Parser/layout-parser'>Github Repo</a></p>"

examples = [
    ['examples/data/example-table.jpeg'],
    ['examples/data/paper-image.jpg']
    
]

gr.Interface(lpi, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()