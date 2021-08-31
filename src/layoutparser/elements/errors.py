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