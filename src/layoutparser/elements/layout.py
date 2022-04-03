# Copyright 2021 The Layout sParser team. All rights reserved.
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

from typing import List, Union, Dict, Dict, Any, Optional
from collections.abc import MutableSequence, Iterable
from copy import copy

import pandas as pd

from .base import BaseCoordElement, BaseLayoutElement
from .layout_elements import (
    Interval,
    Rectangle,
    Quadrilateral,
    TextBlock,
    ALL_BASECOORD_ELEMENTS,
    BASECOORD_ELEMENT_NAMEMAP,
    BASECOORD_ELEMENT_INDEXMAP,
)


class Layout(MutableSequence):
    """
    The :obj:`Layout` class id designed for processing a list of layout elements
    on a page. It stores the layout elements in a list and the related `page_data`,
    and provides handy APIs for processing all the layout elements in batch. `

    Args:
        blocks (:obj:`list`):
            A list of layout element blocks
        page_data (Dict, optional):
            A dictionary storing the page (canvas) related information
            like `height`, `width`, etc. It should be passed in as a
            keyword argument to avoid any confusion.
            Defaults to None.
    """

    def __init__(self, blocks: Optional[List] = None, *, page_data: Dict = None):

        if not (
            (blocks is None)
            or (isinstance(blocks, Iterable) and blocks.__class__.__name__ != "Layout")
        ):

            if blocks.__class__.__name__ == "Layout":
                error_msg = f"Please check the input: it should be lp.Layout([layout]) instead of lp.Layout(layout)"
            else:
                error_msg = f"Blocks should be a list of layout elements or empty (None), instead got {blocks}.\n"
            raise ValueError(error_msg)
            
        if isinstance(blocks, tuple):
            blocks = list(blocks) # <- more robust handling for tuple-like inputs

        self._blocks = blocks if blocks is not None else []
        self.page_data = page_data or {}

    def __getitem__(self, key):
        blocks = self._blocks[key]
        if isinstance(key, slice):
            return self.__class__(self._blocks[key], page_data=self.page_data)
        else:
            return blocks

    def __setitem__(self, key, newvalue):
        self._blocks[key] = newvalue

    def __delitem__(self, key):
        del self._blocks[key]

    def __len__(self):
        return len(self._blocks)

    def __iter__(self):
        for ele in self._blocks:
            yield ele

    def __repr__(self):
        info_str = ", ".join([f"{key}={val}" for key, val in vars(self).items()])
        return f"{self.__class__.__name__}({info_str})"

    def __eq__(self, other):
        if isinstance(other, Layout):
            return self._blocks == other._blocks and self.page_data == other.page_data
        else:
            return False

    def __add__(self, other):
        if isinstance(other, Layout):
            if self.page_data == other.page_data:
                return self.__class__(
                    self._blocks + other._blocks, page_data=self.page_data
                )
            elif self.page_data == {} or other.page_data == {}:
                return self.__class__(
                    self._blocks + other._blocks,
                    page_data=self.page_data or other.page_data,
                )
            else:
                raise ValueError(
                    f"Incompatible page_data for two innputs: {self.page_data} vs {other.page_data}."
                )
        elif isinstance(other, list):
            return self.__class__(self._blocks + other, page_data=self.page_data)
        else:
            raise ValueError(
                f"Invalid input type for other {other.__class__.__name__}."
            )

    def insert(self, key, value):
        self._blocks.insert(key, value)

    def copy(self):
        return self.__class__(copy(self._blocks), page_data=self.page_data)

    def relative_to(self, other):
        return self.__class__(
            [ele.relative_to(other) for ele in self], page_data=self.page_data
        )

    def condition_on(self, other):
        return self.__class__(
            [ele.condition_on(other) for ele in self], page_data=self.page_data
        )

    def is_in(self, other, soft_margin={}, center=False):
        return self.__class__(
            [ele.is_in(other, soft_margin, center) for ele in self],
            page_data=self.page_data,
        )

    def sort(self, key=None, reverse=False, inplace=False) -> Optional["Layout"]:
        """Sort the list of blocks based on the given

        Args:
            key ([type], optional): key specifies a function of one argument that
            is used to extract a comparison key from each list element.
            Defaults to None.
            reverse (bool, optional): reverse is a boolean value. If set to True,
            then the list elements are sorted as if each comparison were reversed.
            Defaults to False.
            inplace (bool, optional): whether to perform the sort inplace. If set
            to False, it will return another object instance with _block sorted in
            the order. Defaults to False.

        Examples::
            >>> import layoutparser as lp
            >>> i = lp.Interval(4, 5, axis="y")
            >>> l = lp.Layout([i, i.shift(2)])
            >>> l.sort(key=lambda x: x.coordinates[1], reverse=True)

        """
        if not inplace:
            return self.__class__(
                sorted(self._blocks, key=key, reverse=reverse), page_data=self.page_data
            )
        else:
            self._blocks.sort(key=key, reverse=reverse)

    def filter_by(self, other, soft_margin={}, center=False):
        """
        Return a `Layout` object containing the elements that are in the `other` object.

        Args:
            other (:obj:`BaseCoordElement`):
                The block to filter the current elements.

        Returns:
            :obj:`Layout`:
                A new layout object after filtering.
        """
        return self.__class__(
            [ele for ele in self if ele.is_in(other, soft_margin, center)],
            page_data=self.page_data,
        )

    def shift(self, shift_distance):
        """
        Shift all layout elements by user specified amounts on x and y axis respectively. If shift_distance is one
        numeric value, the element will by shifted by the same specified amount on both x and y axis.

        Args:
            shift_distance (:obj:`numeric` or :obj:`Tuple(numeric)` or :obj:`List[numeric]`):
                The number of pixels used to shift the element.

        Returns:
            :obj:`Layout`:
                A new layout object with all the elements shifted in the specified values.
        """
        return self.__class__(
            [ele.shift(shift_distance) for ele in self], page_data=self.page_data
        )

    def pad(self, left=0, right=0, top=0, bottom=0, safe_mode=True):
        """Pad all layout elements on the four sides of the polygon with the user-defined pixels. If
        safe_mode is set to True, the function will cut off the excess padding that falls on the negative
        side of the coordinates.

        Args:
            left (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the upper side of the polygon.
            right (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the lower side of the polygon.
            top (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the left side of the polygon.
            bottom (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the right side of the polygon.
            safe_mode (:obj:`bool`, `optional`, defaults to True): A bool value to toggle the safe_mode.

        Returns:
            :obj:`Layout`:
                A new layout object with all the elements padded in the specified values.
        """
        return self.__class__(
            [ele.pad(left, right, top, bottom, safe_mode) for ele in self],
            page_data=self.page_data,
        )

    def scale(self, scale_factor):
        """
        Scale all layout element by a user specified amount on x and y axis respectively. If scale_factor is one
        numeric value, the element will by scaled by the same specified amount on both x and y axis.

        Args:
            scale_factor (:obj:`numeric` or :obj:`Tuple(numeric)` or :obj:`List[numeric]`): The amount for downscaling or upscaling the element.

        Returns:
            :obj:`Layout`:
                A new layout object with all the elements scaled in the specified values.
        """
        return self.__class__(
            [ele.scale(scale_factor) for ele in self], page_data=self.page_data
        )

    def crop_image(self, image):
        return [ele.crop_image(image) for ele in self]

    def get_texts(self):
        """
        Iterate through all the text blocks in the list and append their ocr'ed text results.

        Returns:
            :obj:`List[str]`: A list of text strings of the text blocks in the list of layout elements.
        """

        return [ele.text for ele in self if hasattr(ele, "text")]

    def get_info(self, attr_name):
        """Given user-provided attribute name, check all the elements in the list and return the corresponding
        attribute values.

        Args:
            attr_name (:obj:`str`): The text string of certain attribute name.

        Returns:
            :obj:`List`:
                The list of the corresponding attribute value (if exist) of each element in the list.
        """
        return [getattr(ele, attr_name) for ele in self if hasattr(ele, attr_name)]

    def to_dict(self) -> Dict[str, Any]:
        """Generate a dict representation of the layout object with
        the page_data and all the blocks in its dict representation.

        Returns:
            :obj:`Dict`:
                The dictionary representation of the layout object.
        """
        return {"page_data": self.page_data, "blocks": [ele.to_dict() for ele in self]}

    def get_homogeneous_blocks(self) -> List[BaseLayoutElement]:
        """Convert all elements into blocks of the same type based
        on the type casting rule::

            Interval < Rectangle < Quadrilateral < TextBlock

        Returns:
            List[BaseLayoutElement]:
                A list of base layout elements of the maximal compatible
                type
        """

        # Detect the maximal compatible type
        has_textblock = False
        max_coord_level = -1
        for ele in self:

            if isinstance(ele, TextBlock):
                has_textblock = True
                block = ele.block
            else:
                block = ele

            max_coord_level = max(
                max_coord_level, BASECOORD_ELEMENT_INDEXMAP[block._name]
            )
        target_coord_name = ALL_BASECOORD_ELEMENTS[max_coord_level]._name

        if has_textblock:
            new_blocks = []
            for ele in self:
                if isinstance(ele, TextBlock):
                    ele = copy(ele)
                    if ele.block._name != target_coord_name:
                        ele.block = getattr(ele.block, f"to_{target_coord_name}")()
                else:
                    if ele._name != target_coord_name:
                        ele = getattr(ele, f"to_{target_coord_name}")()
                    ele = TextBlock(block)
                new_blocks.append(ele)
        else:
            new_blocks = [
                getattr(ele, f"to_{target_coord_name}")()
                if ele._name != target_coord_name
                else ele
                for ele in self
            ]

        return new_blocks

    def to_dataframe(self, enforce_same_type=False) -> pd.DataFrame:
        """Convert the layout object into the dataframe.
        Warning: the page data won't be exported.

        Args:
            enforce_same_type (:obj:`bool`, optional):
                If true, it will convert all the contained blocks to
                the maximal compatible data type.
                Defaults to False.

        Returns:
            pd.DataFrame:
                The dataframe representation of layout object
        """
        if enforce_same_type:
            blocks = self.get_homogeneous_blocks()
        else:
            blocks = self

        df = pd.DataFrame([ele.to_dict() for ele in blocks])

        return df
