# Copyright 2021 The Layout Parser team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union, Dict, Any, Tuple
import functools
import os
import sys
from itertools import cycle

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageColor

import layoutparser
from .elements import (
    Layout,
    Interval,
    Rectangle,
    TextBlock,
    Quadrilateral,
)
from .elements.utils import cvt_coordinates_to_points

# We need to fix this ugly hack some time in the future
_lib_path = os.path.dirname(sys.modules[layoutparser.__package__].__file__)
_font_path = os.path.join(_lib_path, "misc", "NotoSerifCJKjp-Regular.otf")

DEFAULT_BOX_WIDTH_RATIO = 0.005
DEFAULT_OUTLINE_COLOR = "red"
DEAFULT_COLOR_PALETTE = "#f6bd60-#f7ede2-#f5cac3-#84a59d-#f28482"
# From https://coolors.co/f6bd60-f7ede2-f5cac3-84a59d-f28482

DEFAULT_FONT_PATH = _font_path
DEFAULT_FONT_SIZE = 15
DEFAULT_FONT_OBJECT = ImageFont.truetype(DEFAULT_FONT_PATH, DEFAULT_FONT_SIZE)
DEFAULT_TEXT_COLOR = "black"
DEFAULT_TEXT_BACKGROUND = "white"

__all__ = ["draw_box", "draw_text"]


def _draw_vertical_text(
    text,
    image_font,
    text_color,
    text_background_color,
    character_spacing=2,
    space_width=1,
):
    """Helper function to draw text vertically.
    Ref: https://github.com/Belval/TextRecognitionDataGenerator/blob/7f4c782c33993d2b6f712d01e86a2f342025f2df/trdg/computer_text_generator.py
    """

    space_height = int(image_font.getsize(" ")[1] * space_width)

    char_heights = [
        image_font.getsize(c)[1] if c != " " else space_height for c in text
    ]
    text_width = max([image_font.getsize(c)[0] for c in text])
    text_height = sum(char_heights) + character_spacing * len(text)

    txt_img = Image.new("RGBA", (text_width, text_height), color=text_background_color)
    txt_mask = Image.new("RGBA", (text_width, text_height), color=text_background_color)

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
            font_path or DEFAULT_FONT_PATH, font_size or DEFAULT_FONT_SIZE
        )


def _create_new_canvas(canvas, arrangement, text_background_color):

    if arrangement == "lr":
        new_canvas = Image.new(
            "RGBA",
            (canvas.width * 2, canvas.height),
            color=text_background_color or DEFAULT_TEXT_BACKGROUND,
        )
        new_canvas.paste(canvas, (canvas.width, 0))

    elif arrangement == "ud":
        new_canvas = Image.new(
            "RGBA",
            (canvas.width, canvas.height * 2),
            color=text_background_color or DEFAULT_TEXT_BACKGROUND,
        )
        new_canvas.paste(canvas, (0, canvas.height))

    else:
        raise ValueError(f"Invalid direction {arrangement}")

    return new_canvas


def _create_color_palette(types):
    return {
        type: color
        for type, color in zip(types, cycle(DEAFULT_COLOR_PALETTE.split("-")))
    }


def _get_color_rgb(color_string: Any, alpha: float) -> Tuple[int, int, int, int]:
    if color_string[0] == "#" and len(color_string) == 7:
        # When color string is a hex string
        color_hex = color_string.lstrip("#")
        return (
            *tuple(int(color_hex[i : i + 2], 16) for i in (0, 2, 4)),
            int(255 * alpha),
        )
    else:
        try:
            rgb = ImageColor.getrgb(color_string)
            return rgb + (int(255 * alpha),)
        except:
            # ImageColor.getrgb will throw an ValueError when the color is not
            # a valid color string, even if it is in other formats supported by
            # PIL. As such, we return the color as it is if the first two cases
            # are not valid.
            return color_string


def _draw_box_outline_on_handler(draw, block, color, width):

    if not hasattr(block, "points"):
        points = (cvt_coordinates_to_points(block.coordinates),)
    else:
        points = block.points

    vertices = points.ravel().tolist()
    drawing_vertices = vertices + vertices[:2]

    draw.line(
        drawing_vertices,
        width=width,
        fill=color,
    )


def _draw_transparent_box_on_handler(draw, block, color, alpha):

    if hasattr(block, "points"):
        vertices = [tuple(block) for block in block.points.tolist()]
    else:
        vertices = cvt_coordinates_to_points(block.coordinates)

    draw.polygon(
        vertices,
        _get_color_rgb(color, alpha),
    )


def image_loader(func):
    @functools.wraps(func)
    def wrap(canvas, layout, *args, **kwargs):

        if isinstance(canvas, Image.Image):
            if canvas.mode != "RGB":
                canvas = canvas.convert("RGB")
            canvas = canvas.copy()
        elif isinstance(canvas, np.ndarray):
            canvas = Image.fromarray(canvas)
        out = func(canvas, layout, *args, **kwargs)
        return out

    return wrap


@image_loader
def draw_transparent_box(
    canvas: "Image",
    blocks: Layout,
    color_map: Dict = None,
    alpha: float = 0.25,
) -> "Image":
    """Given the image, draw a series of transparent boxes based on the blocks,
    coloring using the specified color_map.
    """

    if color_map is None:
        all_types = set([b.type for b in blocks if hasattr(b, "type")])
        color_map = _create_color_palette(all_types)

    canvas = canvas.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")

    for block in blocks:
        _draw_transparent_box_on_handler(draw, block, color_map[block.type], alpha)

    return canvas


@image_loader
def draw_box(
    canvas,
    layout,
    box_width=None,
    box_alpha=0,
    color_map=None,
    show_element_id=False,
    show_element_type=False,
    id_font_size=None,
    id_font_path=None,
    id_text_color=None,
    id_text_background_color=None,
    id_text_background_alpha=1,
):
    """Draw the layout region on the input canvas(image).

    Args:
        canvas (:obj:`~np.ndarray` or :obj:`~PIL.Image.Image`):
            The canvas to draw the layout boxes.
        layout (:obj:`Layout` or :obj:`list`):
            The layout of the canvas to show.
        box_width (:obj:`int`, optional):
            Set to change the width of the drawn layout box boundary.
            Defaults to None, when the boundary is automatically
            calculated as the the :const:`DEFAULT_BOX_WIDTH_RATIO`
            * the maximum of (height, width) of the canvas.
        box_alpha (:obj:`float`, optional):
            A float range from 0 to 1. Set to change the alpha of the
            drawn layout box.
            Defaults to 0 - the layout box will be fully transparent.
        color_map (dict, optional):
            A map from `block.type` to the colors, e.g., `{1: 'red'}`.
            You can set it to `{}` to use only the
            :const:`DEFAULT_OUTLINE_COLOR` for the outlines.
            Defaults to None, when a color palette is is automatically
            created based on the input layout.
        show_element_id (bool, optional):
            Whether to display `block.id` on the top-left corner of
            the block.
            Defaults to False.
        show_element_type (bool, optional):
            Whether to display `block.type` on the top-left corner of
            the block.
            Defaults to False.
        id_font_size (int, optional):
            Set to change the font size used for drawing `block.id`.
            Defaults to None, when the size is set to
            :const:`DEFAULT_FONT_SIZE`.
        id_font_path (:obj:`str`, optional):
            Set to change the font used for drawing `block.id`.
            Defaults to None, when the :const:`DEFAULT_FONT_OBJECT` is used.
        id_text_color (:obj:`str`, optional):
            Set to change the text color used for drawing `block.id`.
            Defaults to None, when the color is set to
            :const:`DEFAULT_TEXT_COLOR`.
        id_text_background_color (:obj:`str`, optional):
            Set to change the text region background used for drawing `block.id`.
            Defaults to None, when the color is set to
            :const:`DEFAULT_TEXT_BACKGROUND`.
        id_text_background_alpha (:obj:`float`, optional):
            A float range from 0 to 1. Set to change the alpha of the
            drawn text.
            Defaults to 1 - the text box will be solid.
    Returns:
        :obj:`PIL.Image.Image`:
            A Image object containing the `layout` draw upon the input `canvas`.
    """

    assert 0 <= box_alpha <= 1, ValueError(
        f"The box_alpha value {box_alpha} is not within range [0,1]."
    )
    assert 0 <= id_text_background_alpha <= 1, ValueError(
        f"The id_text_background_alpha value {id_text_background_alpha} is not within range [0,1]."
    )

    draw = ImageDraw.Draw(canvas, mode="RGBA")

    id_text_background_color = id_text_background_color or DEFAULT_TEXT_BACKGROUND
    id_text_color = id_text_color or DEFAULT_TEXT_COLOR

    if box_width is None:
        box_width = _calculate_default_box_width(canvas)

    if show_element_id or show_element_type:
        font_obj = _create_font_object(id_font_size, id_font_path)

    if color_map is None:
        all_types = set([b.type for b in layout if hasattr(b, "type")])
        color_map = _create_color_palette(all_types)

    for idx, ele in enumerate(layout):

        if isinstance(ele, Interval):
            ele = ele.put_on_canvas(canvas)

        outline_color = (
            DEFAULT_OUTLINE_COLOR
            if not isinstance(ele, TextBlock)
            else color_map.get(ele.type, DEFAULT_OUTLINE_COLOR)
        )

        _draw_box_outline_on_handler(draw, ele, outline_color, box_width)

        _draw_transparent_box_on_handler(draw, ele, outline_color, box_alpha)

        if show_element_id or show_element_type:
            text = ""
            if show_element_id:
                ele_id = ele.id or idx
                text += str(ele_id)
            if show_element_type:
                text = str(ele.type) if not text else text + ": " + str(ele.type)

            start_x, start_y = ele.coordinates[:2]
            text_w, text_h = font_obj.getsize(text)

            text_box_object = Rectangle(
                start_x, start_y, start_x + text_w, start_y + text_h
            )
            # Add a small background for the text

            _draw_transparent_box_on_handler(
                draw,
                text_box_object,
                id_text_background_color,
                id_text_background_alpha,
            )

            # Draw the ids
            draw.text(
                (start_x, start_y),
                text,
                fill=id_text_color,
                font=font_obj,
            )

    return canvas


@image_loader
def draw_text(
    canvas,
    layout,
    arrangement="lr",
    font_size=None,
    font_path=None,
    text_color=None,
    text_background_color=None,
    text_background_alpha=1,
    vertical_text=False,
    with_box_on_text=False,
    text_box_width=None,
    text_box_color=None,
    text_box_alpha=0,
    with_layout=False,
    **kwargs,
):
    """Draw the (detected) text in the `layout` according to
    their coordinates next to the input `canvas` (image) for better comparison.

    Args:
        canvas (:obj:`~np.ndarray` or :obj:`~PIL.Image.Image`):
            The canvas to draw the layout boxes.
        layout (:obj:`Layout` or :obj:`list`):
            The layout of the canvas to show.
        arrangement (`{'lr', 'ud'}`, optional):
            The arrangement of the drawn text canvas and the original
            image canvas:
            * `lr` - left and right
            * `ud` - up and down

            Defaults to 'lr'.
        font_size (:obj:`str`, optional):
            Set to change the size of the font used for
            drawing `block.text`.
            Defaults to None, when the size is set to
            :const:`DEFAULT_FONT_SIZE`.
        font_path (:obj:`str`, optional):
            Set to change the font used for drawing `block.text`.
            Defaults to None, when the :const:`DEFAULT_FONT_OBJECT` is used.
        text_color ([type], optional):
            Set to change the text color used for drawing `block.text`.
            Defaults to None, when the color is set to
            :const:`DEFAULT_TEXT_COLOR`.
        text_background_color ([type], optional):
            Set to change the text region background used for drawing
            `block.text`.
            Defaults to None, when the color is set to
            :const:`DEFAULT_TEXT_BACKGROUND`.
        text_background_alpha (:obj:`float`, optional):
            A float range from 0 to 1. Set to change the alpha of the
            background of the canvas.
            Defaults to 1 - the text box will be solid.
        vertical_text (bool, optional):
            Whether the text in a block should be drawn vertically.
            Defaults to False.
        with_box_on_text (bool, optional):
            Whether to draw the layout box boundary of a text region
            on the text canvas.
            Defaults to False.
        text_box_width (:obj:`int`, optional):
            Set to change the width of the drawn layout box boundary.
            Defaults to None, when the boundary is automatically
            calculated as the the :const:`DEFAULT_BOX_WIDTH_RATIO`
            * the maximum of (height, width) of the canvas.
        text_box_alpha (:obj:`float`, optional):
            A float range from 0 to 1. Set to change the alpha of the
            drawn text box.
            Defaults to 0 - the text box will be fully transparent.
        text_box_color (:obj:`int`, optional):
            Set to change the color of the drawn layout box boundary.
            Defaults to None, when the color is set to
            :const:`DEFAULT_OUTLINE_COLOR`.
        
        with_layout (bool, optional):
            Whether to draw the layout boxes on the input (image) canvas.
            Defaults to False.
            When set to true, you can pass in the arguments in
            :obj:`draw_box` to change the style of the drawn layout boxes.

    Returns:
        :obj:`PIL.Image.Image`:
            A Image object containing the drawn text from `layout`.
    """

    assert 0 <= text_background_alpha <= 1, ValueError(
        f"The text_background_color value {text_background_alpha} is not within range [0,1]."
    )
    assert 0 <= text_box_alpha <= 1, ValueError(
        f"The text_box_alpha value {text_box_alpha} is not within range [0,1]."
    )

    if with_box_on_text:
        if text_box_width is None:
            text_box_width = _calculate_default_box_width(canvas)

    if with_layout:
        canvas = draw_box(canvas, layout, **kwargs)

    font_obj = _create_font_object(font_size, font_path)
    text_box_color = text_box_color or DEFAULT_OUTLINE_COLOR
    text_color = text_color or DEFAULT_TEXT_COLOR
    text_background_color = text_background_color or DEFAULT_TEXT_BACKGROUND

    canvas = _create_new_canvas(
        canvas,
        arrangement,
        _get_color_rgb(text_background_color, text_background_alpha),
    )
    draw = ImageDraw.Draw(canvas, "RGBA")

    for idx, ele in enumerate(layout):

        if with_box_on_text:
            modified_box = ele.pad(right=text_box_width, bottom=text_box_width)

            _draw_box_outline_on_handler(
                draw, modified_box, text_box_color, text_box_width
            )
            _draw_transparent_box_on_handler(
                draw, modified_box, text_box_color, text_box_alpha
            )

        if not hasattr(ele, "text") or ele.text == "":
            continue

        (start_x, start_y) = ele.coordinates[:2]
        if not vertical_text:
            draw.text(
                (start_x, start_y),
                ele.text,
                font=font_obj,
                fill=text_color,
            )
        else:
            text_segment = _draw_vertical_text(
                ele.text,
                font_obj,
                text_color,
                _get_color_rgb(text_background_color, text_background_alpha),
            )

            if with_box_on_text:
                # Avoid cover the box regions
                canvas.paste(
                    text_segment, (start_x + text_box_width, start_y + text_box_width)
                )
            else:
                canvas.paste(text_segment, (start_x, start_y))

    return canvas
