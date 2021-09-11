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

from typing import List, Dict, Dict, Any
from abc import ABC, abstractmethod
from copy import copy

class BaseLayoutElement:
    def set(self, inplace=False, **kwargs):

        obj = self if inplace else copy(self)
        var_dict = vars(obj)
        for key, val in kwargs.items():
            if key in var_dict:
                var_dict[key] = val
            elif f"_{key}" in var_dict:
                var_dict[f"_{key}"] = val
            else:
                raise ValueError(f"Unknown attribute name: {key}")

        return obj

    def __repr__(self):

        info_str = ", ".join([f"{key}={val}" for key, val in vars(self).items()])
        return f"{self.__class__.__name__}({info_str})"

    def __eq__(self, other):

        if other.__class__ is not self.__class__:
            return False

        return vars(self) == vars(other)


class BaseCoordElement(ABC, BaseLayoutElement):
    @property
    @abstractmethod
    def _name(self) -> str:
        """The name of the class"""
        pass

    @property
    @abstractmethod
    def _features(self) -> List[str]:
        """A list of features names used for initializing the class object"""
        pass

    #######################################################################
    #########################  Layout Properties  #########################
    #######################################################################

    @property
    @abstractmethod
    def width(self):
        pass

    @property
    @abstractmethod
    def height(self):
        pass

    @property
    @abstractmethod
    def coordinates(self):
        pass

    @property
    @abstractmethod
    def points(self):
        pass

    @property
    @abstractmethod
    def area(self):
        pass

    #######################################################################
    ###   Geometric Relations (relative to, condition on, and is in)    ###
    #######################################################################

    @abstractmethod
    def condition_on(self, other):
        """
        Given the current element in relative coordinates to another element which is in absolute coordinates,
        generate a new element of the current element in absolute coordinates.

        Args:
            other (:obj:`BaseCoordElement`):
                The other layout element involved in the geometric operations.

        Raises:
            Exception: Raise error when the input type of the other element is invalid.

        Returns:
            :obj:`BaseCoordElement`:
                The BaseCoordElement object of the original element in the absolute coordinate system.
        """

        pass

    @abstractmethod
    def relative_to(self, other):
        """
        Given the current element and another element both in absolute coordinates,
        generate a new element of the current element in relative coordinates to the other element.

        Args:
            other (:obj:`BaseCoordElement`): The other layout element involved in the geometric operations.

        Raises:
            Exception: Raise error when the input type of the other element is invalid.

        Returns:
            :obj:`BaseCoordElement`:
                The BaseCoordElement object of the original element in the relative coordinate system.
        """

        pass

    @abstractmethod
    def is_in(self, other, soft_margin={}, center=False):
        """
        Identify whether the current element is within another element.

        Args:
            other (:obj:`BaseCoordElement`):
                The other layout element involved in the geometric operations.
            soft_margin (:obj:`dict`, `optional`, defaults to `{}`):
                Enlarge the other element with wider margins to relax the restrictions.
            center (:obj:`bool`, `optional`, defaults to `False`):
                The toggle to determine whether the center (instead of the four corners)
                of the current element is in the other element.

        Returns:
            :obj:`bool`: Returns `True` if the current element is in the other element and `False` if not.
        """

        pass

    #######################################################################
    ################# Shape Operations (intersect, union)  ################
    #######################################################################

    @abstractmethod
    def intersect(self, other: "BaseCoordElement", strict: bool = True):
        """Intersect the current shape with the other object, with operations defined in
        :doc:`../notes/shape_operations`.
        """

    @abstractmethod
    def union(self, other: "BaseCoordElement", strict: bool = True):
        """Union the current shape with the other object, with operations defined in
        :doc:`../notes/shape_operations`.
        """

    #######################################################################
    ############### Geometric Operations (pad, shift, scale) ##############
    #######################################################################

    @abstractmethod
    def pad(self, left=0, right=0, top=0, bottom=0, safe_mode=True):
        """Pad the layout element on the four sides of the polygon with the user-defined pixels. If
        safe_mode is set to True, the function will cut off the excess padding that falls on the negative
        side of the coordinates.

        Args:
            left (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the upper side of the polygon.
            right (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the lower side of the polygon.
            top (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the left side of the polygon.
            bottom (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the right side of the polygon.
            safe_mode (:obj:`bool`, `optional`, defaults to True): A bool value to toggle the safe_mode.

        Returns:
            :obj:`BaseCoordElement`: The padded BaseCoordElement object.
        """

        pass

    @abstractmethod
    def shift(self, shift_distance=0):
        """
        Shift the layout element by user specified amounts on x and y axis respectively. If shift_distance is one
        numeric value, the element will by shifted by the same specified amount on both x and y axis.

        Args:
            shift_distance (:obj:`numeric` or :obj:`Tuple(numeric)` or :obj:`List[numeric]`):
                The number of pixels used to shift the element.

        Returns:
            :obj:`BaseCoordElement`: The shifted BaseCoordElement of the same shape-specific class.
        """

        pass

    @abstractmethod
    def scale(self, scale_factor=1):
        """
        Scale the layout element by a user specified amount on x and y axis respectively. If scale_factor is one
        numeric value, the element will by scaled by the same specified amount on both x and y axis.

        Args:
            scale_factor (:obj:`numeric` or :obj:`Tuple(numeric)` or :obj:`List[numeric]`): The amount for downscaling or upscaling the element.

        Returns:
            :obj:`BaseCoordElement`: The scaled BaseCoordElement of the same shape-specific class.
        """

        pass

    #######################################################################
    ################################# MISC ################################
    #######################################################################

    @abstractmethod
    def crop_image(self, image):
        """
        Crop the input image according to the coordinates of the element.

        Args:
            image (:obj:`Numpy array`): The array of the input image.

        Returns:
            :obj:`Numpy array`: The array of the cropped image.
        """

        pass

    #######################################################################
    ########################## Import and Export ##########################
    #######################################################################

    def to_dict(self) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the current object:
            {
                "block_type": <"interval", "rectangle", "quadrilateral"> ,
                "non_empty_block_attr1": value1,
                ...
            }
        """

        data = {
            key: getattr(self, key)
            for key in self._features
            if getattr(self, key) is not None
        }
        data["block_type"] = self._name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseCoordElement":
        """Initialize an instance based on the dictionary representation

        Args:
            data (:obj:`dict`): The dictionary representation of the object
        """

        assert (
            cls._name == data["block_type"]
        ), f"Incompatible block types {data['block_type']}"

        return cls(**{f: data[f] for f in cls._features})

