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

from typing import List, Union, Optional, Dict, Tuple

import pdfplumber
import pandas as pd

from ..elements import Layout
from .basic import load_dataframe

DEFAULT_PDF_DPI = 72


def extract_words_for_page(
    page: pdfplumber.page.Page,
    x_tolerance=1.5,
    y_tolerance=2,
    keep_blank_chars=False,
    use_text_flow=True,
    horizontal_ltr=True,
    vertical_ttb=True,
    extra_attrs=None,
) -> Layout:
    """The helper function used for extracting words from a pdfplumber page
    object. 

    Returns:
        Layout: a layout object representing all extracted pdf tokens on this page. 
    """
    if extra_attrs is None:
        extra_attrs = ["fontname", "size"]

    tokens = page.extract_words(
        x_tolerance=x_tolerance,
        y_tolerance=y_tolerance,
        keep_blank_chars=keep_blank_chars,
        use_text_flow=use_text_flow,
        horizontal_ltr=horizontal_ltr,
        vertical_ttb=vertical_ttb,
        extra_attrs=extra_attrs,
    )

    df = pd.DataFrame(tokens)
    
    if len(df) == 0:
        return Layout()
    
    df[["x0", "x1"]] = (
        df[["x0", "x1"]].clip(lower=0, upper=int(page.width)).astype("float")
    )
    df[["top", "bottom"]] = (
        df[["top", "bottom"]].clip(lower=0, upper=int(page.height)).astype("float")
    )

    page_tokens = load_dataframe(
        df.reset_index().rename(
            columns={
                "x0": "x_1",
                "x1": "x_2",
                "top": "y_1",
                "bottom": "y_2",
                "index": "id",
                "fontname": "type",  # also loading fontname as "type"
            }
        ),
        block_type="rectangle",
    )

    return page_tokens


def load_pdf(
    filename: str,
    load_images: bool = False,
    x_tolerance: int = 1.5,
    y_tolerance: int = 2,
    keep_blank_chars: bool = False,
    use_text_flow: bool = True,
    horizontal_ltr: bool = True,
    vertical_ttb: bool = True,
    extra_attrs: Optional[List[str]] = None,
    dpi: int = DEFAULT_PDF_DPI,
) -> Union[List[Layout], Tuple[List[Layout], List["Image.Image"]]]:
    """Load all tokens for each page from a PDF file, and save them
    in a list of Layout objects with the original page order.

    Args:
        filename (str): The path to the PDF file.
        load_images (bool, optional):
            Whether load screenshot for each page of the PDF file.
            When set to true, the function will return both the layout and
            screenshot image for each page.
            Defaults to False.
        x_tolerance (int, optional):
            The threshold used for extracting "word tokens" from the pdf file.
            It will merge the pdf characters into a word token if the difference
            between the x_2 of one character and the x_1 of the next is less than
            or equal to x_tolerance. See details in `pdf2plumber's documentation
            <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
            Defaults to 1.5.
        y_tolerance (int, optional):
            The threshold used for extracting "word tokens" from the pdf file.
            It will merge the pdf characters into a word token if the difference
            between the y_2 of one character and the y_1 of the next is less than
            or equal to y_tolerance. See details in `pdf2plumber's documentation
            <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
            Defaults to 2.
        keep_blank_chars (bool, optional):
            When keep_blank_chars is set to True, it will treat blank characters
            are treated as part of a word, not as a space between words. See
            details in `pdf2plumber's documentation
            <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
            Defaults to False.
        use_text_flow (bool, optional):
            When use_text_flow is set to True, it will use the PDF's underlying
            flow of characters as a guide for ordering and segmenting the words,
            rather than presorting the characters by x/y position. (This mimics
            how dragging a cursor highlights text in a PDF; as with that, the
            order does not always appear to be logical.) See details in
            `pdf2plumber's documentation
            <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
            Defaults to True.
        horizontal_ltr (bool, optional):
            When horizontal_ltr is set to True, it means the doc should read
            text from left to right, vice versa.
            Defaults to True.
        vertical_ttb (bool, optional):
            When vertical_ttb is set to True, it means the doc should read
            text from top to bottom, vice versa.
            Defaults to True.
        extra_attrs (Optional[List[str]], optional):
            Passing a list of extra_attrs (e.g., ["fontname", "size"]) will
            restrict each words to characters that share exactly the same
            value for each of those `attributes extracted by pdfplumber
            <https://github.com/jsvine/pdfplumber/blob/develop/README.md#char-properties>`_,
            and the resulting word dicts will indicate those attributes.
            See details in `pdf2plumber's documentation
            <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
            Defaults to `["fontname", "size"]`.
        dpi (int, optional):
            When loading images of the pdf, you can also specify the resolution
            (or `DPI, dots per inch <https://en.wikipedia.org/wiki/Dots_per_inch>`_)
            for rendering the images. Higher DPI values mean clearer images (also
            larger file sizes).
            Setting dpi will also automatically resizes the extracted pdf_layout
            to match the sizes of the images. Therefore, when visualizing the
            pdf_layouts, it can be rendered appropriately.
            Defaults to `DEFAULT_PDF_DPI=72`, which is also the default rendering dpi
            from the pdfplumber PDF parser.

    Returns:
        List[Layout]:
            When `load_images=False`, it will only load the pdf_tokens from
            the PDF file. Each element of the list denotes all the tokens appeared
            on a single page, and the list is ordered the same as the original PDF
            page order.
        Tuple[List[Layout], List["Image.Image"]]:
            When `load_images=True`, besides the `all_page_layout`, it will also
            return a list of page images.

    Examples::
        >>> import layoutparser as lp
        >>> pdf_layout = lp.load_pdf("path/to/pdf")
        >>> pdf_layout[0] # the layout for page 0
        >>> pdf_layout, pdf_images = lp.load_pdf("path/to/pdf", load_images=True)
        >>> lp.draw_box(pdf_images[0], pdf_layout[0])
    """

    plumber_pdf_object = pdfplumber.open(filename)

    all_page_layout = []
    for page_id in range(len(plumber_pdf_object.pages)):
        cur_page = plumber_pdf_object.pages[page_id]

        page_tokens = extract_words_for_page(
            cur_page,
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
            keep_blank_chars=keep_blank_chars,
            use_text_flow=use_text_flow,
            horizontal_ltr=horizontal_ltr,
            vertical_ttb=vertical_ttb,
            extra_attrs=extra_attrs,
        )

        # Adding metadata for the current page
        page_tokens.page_data["width"] = float(cur_page.width)
        page_tokens.page_data["height"] = float(cur_page.height)
        page_tokens.page_data["index"] = page_id
        
        all_page_layout.append(page_tokens)

    if not load_images:
        return all_page_layout
    else:
        import pdf2image

        pdf_images = pdf2image.convert_from_path(filename, dpi=dpi)

        for page_id, page_image in enumerate(pdf_images):
            image_width, image_height = page_image.size
            page_layout = all_page_layout[page_id]
            layout_width = page_layout.page_data["width"]
            layout_height = page_layout.page_data["height"]
            if image_width != layout_width or image_height != layout_height:
                scale_x = image_width / layout_width
                scale_y = image_height / layout_height
                page_layout = page_layout.scale((scale_x, scale_y))
                page_layout.page_data["width"] = image_width
                page_layout.page_data["height"] = image_height
                all_page_layout[page_id] = page_layout

        return all_page_layout, pdf_images