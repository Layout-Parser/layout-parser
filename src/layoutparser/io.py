import json
from typing import List, Union, Dict, Dict, Any

from .elements import (
    BaseCoordElement,
    BaseLayoutElement,
    Interval,
    Rectangle,
    Quadrilateral,
    TextBlock,
    Layout,
    BASECOORD_ELEMENT_NAMEMAP,
)


def load_json(filename: str) -> Union[BaseLayoutElement, Layout]:
    """Load a JSON file and automatically parse its layout data type.

    Args:
        filename (str):
            The name of the JSON file.

    Returns:
        Union[BaseLayoutElement, Layout]:
            Based on the JSON file format, it will automatically parse
            the type of the data and load it accordingly.
    """
    with open(filename, "r") as fp:
        res = json.load(fp)

    return load_dict(res)


def load_dict(data: Union[Dict, List[Dict]]) -> Union[BaseLayoutElement, Layout]:
    """Load a dict of list of dict representations of some layout data,
    automatically parse its type, and save it as any of BaseLayoutElement
    or Layout datatype.

    Args:
        data (Union[Dict, List]):
            A dict of list of dict representations of the layout data

    Raises:
        ValueError:
            If the data format is incompatible with the layout-data-JSON format,
            raise a `ValueError`.
        ValueError:
            If any `block_type` name is not in the available list of layout element
            names defined in `BASECOORD_ELEMENT_NAMEMAP`, raise a `ValueError`.

    Returns:
        Union[BaseLayoutElement, Layout]:
            Based on the dict format, it will automatically parse the type of
            the data and load it accordingly.
    """
    if isinstance(data, dict):
        if "page_data" in data:
            # It is a layout instance
            return Layout(load_dict(data["blocks"]), page_data=data["page_data"])
        else:

            if data["block_type"] not in BASECOORD_ELEMENT_NAMEMAP:
                raise ValueError(f"Invalid block_type {data['block_type']}")

            # Check if it is a textblock
            is_textblock = any(ele in data for ele in TextBlock._features)
            if is_textblock:
                return TextBlock.from_dict(data)
            else:
                return BASECOORD_ELEMENT_NAMEMAP[data["block_type"]].from_dict(data)

    elif isinstance(data, list):
        return [load_dict(ele) for ele in data]

    else:
        raise ValueError(f"Invalid input JSON structure.")