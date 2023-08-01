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

from typing import List, Union, Any, Callable, Iterable
from functools import partial, reduce

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..elements import BaseLayoutElement, TextBlock


def generalized_connected_component_analysis_1d(
    sequence: List[Any],
    scoring_func: Callable[[Any, Any], int],
    aggregation_func: Callable[[List[Any]], Any] = None,
    default_score_value: int = 0,
) -> List[Any]:
    """Perform connected componenet analysis for any 1D sequence based on
    the scoring function and the aggregation function.
    It will generate the adjacency_matrix for the 1D sequence object using
    the provided `scoring_func` and find the connected componenets.
    The `aggregation_func` will be used to aggregate all elements within
    identified components (when not set, it will be the identity function).

    Args:
        sequence (List[Any]):
            The provided 1D sequence of objects.
        scoring_func (Callable[[Any, Any], int]):
            The scoring function used to construct the adjacency_matrix.
            It should take two objects in the sequence and produe a integer.
        aggregation_func (Callable[[List[Any]], Any], optional):
            The function used to aggregate the elements within an identified
            component.
            Defaults to the identify function: `lambda x: x`.
        default_score_value (int, optional):
            Used to set the default (background) score values that should be
            not considered when running connected component analysis.
            Defaults to 0.

    Returns:
        List[Any]: A list of length n - the number of the detected componenets.
    """

    if aggregation_func is None:
        aggregation_func = lambda x: x  # Identity Function

    seq_len = len(sequence)
    adjacency_matrix = np.ones((seq_len, seq_len)) * default_score_value

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            adjacency_matrix[i][j] = scoring_func(sequence[i], sequence[j])

    graph = csr_matrix(adjacency_matrix)
    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )

    grouped_sequence = []
    for comp_idx in range(n_components):
        element_idx = np.where(labels == comp_idx)[0]
        grouped_sequence.append(aggregation_func([sequence[i] for i in element_idx]))

    return grouped_sequence


def simple_line_detection(
    layout: Iterable[BaseLayoutElement], x_tolerance: int = 10, y_tolerance: int = 10
) -> List[BaseLayoutElement]:
    """Perform line detection based on connected component analysis.

    The is_line_wise_close is the scoring function, which returns True
    if the y-difference is smaller than the y_tolerance AND the
    x-difference (the horizontal gap between two boxes) is also smaller
    than the x_tolerance, and False otherwise.

    All the detected components will then be passed into aggregation_func,
    which returns the overall union box of all the elements, or the line
    box.

    Args:
        layout (Iterable):
            A list (or Layout) of BaseLayoutElement
        x_tolerance (int, optional):
            The value used for specifying the maximum allowed y-difference
            when considered whether two tokens are from the same line.
            Defaults to 10.
        y_tolerance (int, optional):
            The value used for specifying the maximum allowed horizontal gap
            when considered whether two tokens are from the same line.
            Defaults to 10.

    Returns:
        List[BaseLayoutElement]: A list of BaseLayoutElement, denoting the line boxes.
    """

    def is_line_wise_close(token_a, token_b, x_tolerance, y_tolerance):
        y_a = token_a.block.center[1]
        y_b = token_b.block.center[1]

        a_left, a_right = token_a.block.coordinates[0::2]
        b_left, b_right = token_b.block.coordinates[0::2]

        return (
            abs(y_a - y_b) <= y_tolerance
            and min(abs(a_left - b_right), abs(a_right - b_left)) <= x_tolerance
        )
        # If the y-difference is smaller than the y_tolerance AND
        # the x-difference (the horizontal gap between two boxes)
        # is also smaller than the x_tolerance threshold, then
        # these two tokens are considered as line-wise close.

    detected_lines = generalized_connected_component_analysis_1d(
        layout,
        scoring_func=partial(
            is_line_wise_close, y_tolerance=x_tolerance, x_tolerance=y_tolerance
        ),
        aggregation_func=lambda seq: reduce(layout[0].__class__.union, seq),
    )

    return detected_lines


def group_textblocks_based_on_category(
    layout: Iterable[TextBlock], union_group: bool = True
) -> Union[List[TextBlock], List[List[TextBlock]]]:
    """Group textblocks based on their category (block.type).

    Args:
        layout (Iterable):
            A list (or Layout) of BaseLayoutElement
        union_group (bool):
            Whether to union the boxes within each group.
            Defaults to True.

    Returns:
        List[TextBlock]: When `union_group=True`, it produces a list of
            TextBlocks, denoting the boundaries of each texblock group.
        List[List[TextBlock]]: When `union_group=False`, it preserves
            the elements within each group for further processing.
    """

    if union_group:
        aggregation_func = lambda seq: reduce(layout[0].__class__.union, seq)
    else:
        aggregation_func = None

    detected_group_boxes = generalized_connected_component_analysis_1d(
        layout,
        scoring_func=lambda a, b: a.type == b.type,
        aggregation_func=aggregation_func,
    )

    return detected_group_boxes
