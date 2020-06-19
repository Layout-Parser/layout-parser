from PIL import Image, ImageFont, ImageDraw
from .elements import *
import numpy as np 
import functools
import os, sys, warnings
import layoutparser

# We need to fix this ugly hack some time in the future
_lib_path = os.path.dirname(sys.modules[layoutparser.__package__].__file__)
_font_path = os.path.join(_lib_path, 'misc', 'NotoSerifCJKjp-Regular.otf')

DEFAULT_BOX_WIDTH_RATIO = 0.005
DEFAULT_OUTLINE_COLOR   = 'red'

DEFAULT_FONT_PATH       = _font_path
DEFAULT_FONT_SIZE       = 15
DEFAULT_FONT_OBJECT     = ImageFont.truetype(DEFAULT_FONT_PATH, DEFAULT_FONT_SIZE)
DEFAULT_TEXT_COLOR      = 'black'
DEFAULT_TEXT_BACKGROUND = 'white'

__all__ = ['draw_box', 'draw_text']

def _draw_vertical_text(text, image_font, 
                        text_color, text_background_color,
                        character_spacing=2, space_width=1):
    """Helper function to draw text vertically. 
    Ref: https://github.com/Belval/TextRecognitionDataGenerator/blob/7f4c782c33993d2b6f712d01e86a2f342025f2df/trdg/computer_text_generator.py
    """

    space_height = int(image_font.getsize(" ")[1] * space_width)

    char_heights = [
        image_font.getsize(c)[1] if c != " " else space_height for c in text
    ]
    text_width = max([image_font.getsize(c)[0] for c in text])
    text_height = sum(char_heights) + character_spacing * len(text)

    txt_img = Image.new("RGB", (text_width, text_height),  color=text_background_color)
    txt_mask = Image.new("RGB", (text_width, text_height), color=text_background_color)

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask)

    for i, c in enumerate(text):
        txt_img_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=text_color,
            font=image_font,
        )

    return txt_img.crop(txt_img.getbbox())

def _calculate_default_box_width(canvas):
    return max(1, int(min(canvas.size) * DEFAULT_BOX_WIDTH_RATIO))

def _create_font_object(font_size=None, font_path=None):
    
    if font_size is None and font_path is None:
        return DEFAULT_FONT_OBJECT
    else:
        return ImageFont.truetype(
            font_path or DEFAULT_FONT_PATH,
            font_size or DEFAULT_FONT_SIZE
        )

def _create_new_canvas(canvas, arrangement, text_background_color):
    
    if arrangement == 'lr':
        new_canvas = Image.new('RGB', (canvas.width*2, canvas.height), 
                                color=text_background_color or DEFAULT_TEXT_BACKGROUND)
        new_canvas.paste(canvas, (canvas.width,0))
        
    elif arrangement == 'ud':
        new_canvas = Image.new('RGB', (canvas.width, canvas.height*2), 
                               color=text_background_color or DEFAULT_TEXT_BACKGROUND)
        new_canvas.paste(canvas, (0, canvas.height))
    
    else:
        raise ValueError(f"Invalid direction {arrangement}")

    return new_canvas    
    
def image_loader(func):
    @functools.wraps(func)
    def wrap(canvas, layout, *args, **kwargs):
        
        if isinstance(canvas, Image.Image):
            canvas = canvas
        elif isinstance(canvas, np.ndarray):
            canvas = Image.fromarray(canvas)
        out = func(canvas, layout, *args, **kwargs)
        return out
    
    return wrap

@image_loader
def draw_box(canvas, layout, 
                box_width=None,
                color_map={},
                show_element_id=False,
                id_font_size=None,
                id_font_path=None,
                id_text_color=None,
                id_text_background_color=None):

    draw = ImageDraw.Draw(canvas)
    
    if box_width is None:
        box_width = _calculate_default_box_width(canvas)
    
    if show_element_id:
        font_obj = _create_font_object(id_font_size, id_font_path)
    
    for idx, ele in enumerate(layout):
        
        if isinstance(ele, Interval):
            ele = ele.put_on_canvas(canvas)
        
        outline_color = DEFAULT_OUTLINE_COLOR if not isinstance(ele, TextBlock) \
                    else color_map.get(ele.type, DEFAULT_OUTLINE_COLOR)
        
        if not isinstance(ele, Quadrilateral):
            draw.rectangle(ele.coordinates, width=box_width, 
                            outline=outline_color)
        
        else:
            p = ele.points.ravel().tolist()
            draw.line(p+p[:2], width=box_width, 
                            fill=outline_color)
        
        if show_element_id:
            ele_id = ele.id or idx
            
            start_x, start_y = ele.coordinates[:2]
            text_w, text_h = font_obj.getsize(f'{ele_id}')
            
            # Add a small background for the text
            draw.rectangle((start_x, start_y, start_x + text_w, start_y + text_h), 
                           fill=id_text_background_color or DEFAULT_TEXT_BACKGROUND)
            
            # Draw the ids
            draw.text((start_x, start_y), f'{ele_id}', 
                        fill=id_text_color or DEFAULT_TEXT_COLOR, 
                        font=font_obj)
    
    return canvas

@image_loader
def draw_text(canvas, layout, 
                arrangement        = 'lr',
                font_size          = None,
                font_path          = None,
                text_color         = None,
                text_background_color = None,
                vertical_text      = False,
                with_box_on_text   = False,
                text_box_width     = None,
                text_box_color     = None,
                with_layout        = False,
                **kwargs
            ):

    if with_box_on_text:
        if text_box_width is None:
            text_box_width = _calculate_default_box_width(canvas)

    if with_layout:
        canvas = draw_box(canvas, layout, **kwargs)
    
    font_obj = _create_font_object(font_size, font_path)
    text_box_color = text_box_color or DEFAULT_OUTLINE_COLOR
    text_color = text_color or DEFAULT_TEXT_COLOR
    text_background_color = text_background_color or DEFAULT_TEXT_BACKGROUND
    
    canvas = _create_new_canvas(canvas, arrangement, text_background_color)
    draw = ImageDraw.Draw(canvas)
    
    for idx, ele in enumerate(layout):

        if with_box_on_text:
            p = ele.pad(right=text_box_width, 
                        bottom=text_box_width).points.ravel().tolist()
            
            draw.line(p+p[:2], 
                      width=text_box_width, 
                      fill=text_box_color)
        
        if not hasattr(ele, 'text') or ele.text == '': continue
        
        (start_x, start_y) = ele.coordinates[:2]
        if not vertical_text:
            draw.text((start_x, start_y), ele.text, 
                      font=font_obj,
                      fill=text_color)
        else:
            text_segment = _draw_vertical_text(ele.text, font_obj, text_color, text_background_color)
            
            if with_box_on_text:
                # Avoid cover the box regions 
                canvas.paste(text_segment, (start_x+text_box_width, start_y+text_box_width))
            else:
                canvas.paste(text_segment, (start_x, start_y))
                
    return canvas