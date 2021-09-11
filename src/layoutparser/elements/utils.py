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

from typing import List, Union, Dict, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image


def cvt_coordinates_to_points(coords: Tuple[float, float, float, float]) -> np.ndarray:

    x_1, y_1, x_2, y_2 = coords
    return np.array(
        [
            [x_1, y_1],  # Top Left
            [x_2, y_1],  # Top Right
            [x_2, y_2],  # Bottom Right
            [x_1, y_2],  # Bottom Left
        ]
    )


def cvt_points_to_coordinates(points: np.ndarray) -> Tuple[float, float, float, float]:
    x_1 = points[:, 0].min()
    y_1 = points[:, 1].min()
    x_2 = points[:, 0].max()
    y_2 = points[:, 1].max()
    return (x_1, y_1, x_2, y_2)


def perspective_transformation(
    M: np.ndarray, points: np.ndarray, is_inv: bool = False
) -> np.ndarray:

    if is_inv:
        M = np.linalg.inv(M)

    src_mid = np.hstack([points, np.ones((points.shape[0], 1))]).T  # 3x4
    dst_mid = np.matmul(M, src_mid)

    dst = (dst_mid / dst_mid[-1]).T[:, :2]  # 4x2

    return dst


def vertice_in_polygon(vertice: np.ndarray, polygon_points: np.ndarray) -> bool:
    # The polygon_points are ordered clockwise

    # The implementation is based on the algorithm from
    # https://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/

    points = polygon_points - vertice  # shift the coordinates origin to the vertice
    edges = np.append(points, points[0:1, :], axis=0)
    return all([np.linalg.det([e1, e2]) >= 0 for e1, e2 in zip(edges, edges[1:])])
    # If the points are ordered clockwise, the det should <=0


def polygon_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Calculate the area of polygons using
    `Shoelace Formula <https://en.wikipedia.org/wiki/Shoelace_formula>`_.

    Args:
        xs (`np.ndarray`): The x coordinates of the points
        ys (`np.ndarray`): The y coordinates of the points
    """

    # Refer to: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    # The formula is equivalent to the original one indicated in the wikipedia
    # page.

    return 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))