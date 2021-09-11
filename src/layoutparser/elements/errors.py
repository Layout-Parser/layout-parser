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

class NotSupportedShapeError(Exception):
    """For now (v0.2), if the created shape might be a polygon (shapes with more than 4 vertices),
    layoutparser will raise NotSupportedShapeError. It is expected to be fixed in the future versions.
    See
    :ref:`shape_operations:problems-related-to-the-quadrilateral-class`.
    """


class InvalidShapeError(Exception):
    """For shape operations like intersection of union, lp will raise the InvalidShapeError when
    invalid shapes are created (e.g., intersecting a rectangle and an interval).
    """