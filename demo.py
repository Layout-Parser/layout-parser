import layoutparser as lp
import gradio as gr

model = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config')

def lpi(img):
  layout = model.detect(img) # You need to load the image somewhere else, e.g., image = cv2.imread(...)
  image = lp.draw_box(img, layout,) # With extra configurations
  return image

inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil",label="Output Image")

title = "Layout Parser"
description = "demo for OpenAI's CLIP. To use it, simply upload your image, or click one of the examples to load them and optionally add a text label seperated by commas to help clip classify the image better. Read more at the links below."
article = "<p style='text-align: center'><a href='https://openai.com/blog/clip/'>CLIP: Connecting Text and Images</a> | <a href='https://github.com/openai/CLIP'>Github Repo</a></p>"


gr.Interface(lpi, inputs, outputs, title=title, description=description, article=article).launch(debug=True)