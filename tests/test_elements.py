import pytest
import numpy as np
import pandas as pd

from layoutparser.elements import Interval, Rectangle, Quadrilateral, TextBlock, Layout


def test_interval():

    i = Interval(1, 2, axis="y", canvas_height=30, canvas_width=400)
    i.to_rectangle()
    i.to_quadrilateral()
    assert i.shift(1) == Interval(2, 3, axis="y", canvas_height=30, canvas_width=400)
    assert i.area == 1 * 400

    i = Interval(1, 2, axis="x")
    assert i.shift([1, 2]) == Interval(2, 3, axis="x")
    assert i.scale([2, 1]) == Interval(2, 4, axis="x")
    assert i.pad(left=10, right=20) == Interval(0, 22, axis="x")  # Test the safe_mode
    assert i.pad(left=10, right=20, safe_mode=False) == Interval(-9, 22, axis="x")
    assert i.area == 0

    img = np.random.randint(12, 24, (40, 40))
    img[:, 10:20] = 0
    i = Interval(5, 11, axis="x")
    assert np.unique(i.crop_image(img)[:, -1]) == np.array([0])


def test_rectangle():

    r = Rectangle(1, 2, 3, 4)
    r.to_interval(axis="x")
    r.to_quadrilateral()
    assert r.pad(left=1, right=5, top=2, bottom=4) == Rectangle(0, 0, 8, 8)
    assert r.shift([1, 2]) == Rectangle(2, 4, 4, 6)
    assert r.shift(1) == Rectangle(2, 3, 4, 5)
    assert r.scale([3, 2]) == Rectangle(3, 4, 9, 8)
    assert r.scale(2) == Rectangle(2, 4, 6, 8)
    assert r.area == 4

    img = np.random.randint(12, 24, (40, 40))
    r.crop_image(img).shape == (2, 2)


def test_quadrilateral():

    points = np.array([[2, 2], [6, 2], [6, 7], [2, 6]])
    q = Quadrilateral(points)
    q.to_interval()
    q.to_rectangle()
    assert q.shift(1) == Quadrilateral(points + 1)
    assert q.shift([1, 2]) == Quadrilateral(points + np.array([1, 2]))
    assert q.scale(2) == Quadrilateral(points * 2)
    assert q.scale([3, 2]) == Quadrilateral(points * np.array([3, 2]))
    assert q.pad(left=1, top=2, bottom=4) == Quadrilateral(
        np.array([[1, 0], [6, 0], [6, 11], [1, 10]])
    )
    assert (
        q.mapped_rectangle_points == np.array([[0, 0], [4, 0], [4, 5], [0, 5]])
    ).all()

    points = np.array([[2, 2], [6, 2], [6, 5], [2, 5]])
    q = Quadrilateral(points)
    img = np.random.randint(2, 24, (30, 20)).astype("uint8")
    img[2:5, 2:6] = 0
    assert np.unique(q.crop_image(img)) == np.array([0])

    q = Quadrilateral(np.array([[-2, 0], [0, 2], [2, 0], [0, -2]]))
    assert q.area == 8.0

    q = Quadrilateral([1, 2, 3, 4, 5, 6, 7, 8])
    assert (q.points == np.array([[1, 2], [3, 4], [5, 6], [7, 8]])).all()

    with pytest.raises(ValueError):
        Quadrilateral([1, 2, 3, 4, 5, 6, 7])  # Incompatible list length

    with pytest.raises(ValueError):
        Quadrilateral(np.array([[2, 2], [6, 2], [6, 5]]))  # Incompatible ndarray shape


def test_interval_relations():

    i = Interval(4, 5, axis="y")
    r = Rectangle(3, 3, 5, 6)
    q = Quadrilateral(np.array([[2, 2], [6, 2], [6, 7], [2, 5]]))

    assert i.is_in(i)
    assert i.is_in(r)
    assert i.is_in(q)

    # convert to absolute then convert back to relative
    assert i.condition_on(i).relative_to(i) == i
    assert i.condition_on(r).relative_to(r) == i.put_on_canvas(r).to_rectangle()
    assert i.condition_on(q).relative_to(q) == i.put_on_canvas(q).to_quadrilateral()

    # convert to relative then convert back to absolute
    assert i.relative_to(i).condition_on(i) == i
    assert i.relative_to(r).condition_on(r) == i.put_on_canvas(r).to_rectangle()
    assert i.relative_to(q).condition_on(q) == i.put_on_canvas(q).to_quadrilateral()


def test_rectangle_relations():

    i = Interval(4, 5, axis="y")
    q = Quadrilateral(np.array([[2, 2], [6, 2], [6, 7], [2, 5]]))
    r = Rectangle(3, 3, 5, 6)

    assert not r.is_in(q)
    assert r.is_in(q, soft_margin={"bottom": 1})
    assert r.is_in(q.to_rectangle())
    assert r.is_in(q.to_interval())

    # convert to absolute then convert back to relative
    assert r.condition_on(i).relative_to(i) == r
    assert r.condition_on(r).relative_to(r) == r
    assert r.condition_on(q).relative_to(q) == r.to_quadrilateral()

    # convert to relative then convert back to absolute
    assert r.relative_to(i).condition_on(i) == r
    assert r.relative_to(r).condition_on(r) == r
    assert r.relative_to(q).condition_on(q) == r.to_quadrilateral()


def test_quadrilateral_relations():

    i = Interval(4, 5, axis="y")
    q = Quadrilateral(np.array([[2, 2], [6, 2], [6, 7], [2, 5]]))
    r = Rectangle(3, 3, 5, 6)

    assert not q.is_in(r)
    assert q.is_in(i, soft_margin={"top": 2, "bottom": 2})
    assert q.is_in(r, soft_margin={"left": 1, "top": 1, "right": 1, "bottom": 1})
    assert q.is_in(q)

    # convert to absolute then convert back to relative
    assert q.condition_on(i).relative_to(i) == q
    assert q.condition_on(r).relative_to(r) == q
    assert q.condition_on(q).relative_to(q) == q

    # convert to relative then convert back to absolute
    assert q.relative_to(i).condition_on(i) == q
    assert q.relative_to(r).condition_on(r) == q
    assert q.relative_to(q).condition_on(q) == q


def test_textblock():

    i = Interval(4, 5, axis="y")
    q = Quadrilateral(np.array([[2, 2], [6, 2], [6, 7], [2, 5]]))
    r = Rectangle(3, 3, 5, 6)

    t = TextBlock(i, id=1, type=2, text="12")
    assert (
        t.relative_to(q).condition_on(q).block == i.put_on_canvas(q).to_quadrilateral()
    )
    t.area
    t = TextBlock(r, id=1, type=2, parent="a")
    assert t.relative_to(i).condition_on(i).block == r
    t.area
    t = TextBlock(q, id=1, type=2, parent="a")
    assert t.relative_to(r).condition_on(r).block == q
    t.area

    # Ensure the operations did not change the object itself
    assert t == TextBlock(q, id=1, type=2, parent="a")
    t1 = TextBlock(q, id=1, type=2, parent="a")
    t2 = TextBlock(i, id=1, type=2, text="12")
    t1.relative_to(t2)
    assert t2.is_in(t1)

    t = TextBlock(q, score=0.2)


def test_layout():
    i = Interval(4, 5, axis="y")
    q = Quadrilateral(np.array([[2, 2], [6, 2], [6, 7], [2, 5]]))
    r = Rectangle(3, 3, 5, 6)
    t = TextBlock(i, id=1, type=2, text="12")

    l = Layout([i, q, r])
    l.get_texts()
    l.condition_on(i)
    l.relative_to(q)
    l.filter_by(t)
    l.is_in(r)
    assert l.get_homogeneous_blocks() == [i.to_quadrilateral(), q, r.to_quadrilateral()]

    i2 = TextBlock(i, id=1, type=2, text="12")
    r2 = TextBlock(r, id=1, type=2, parent="a")
    q2 = TextBlock(q, id=1, type=2, next="a")
    l2 = Layout([i2, r2, q2], page_data={"width": 200, "height": 200})
    
    l2.get_texts()
    l2.get_info("next")
    l2.condition_on(i)
    l2.relative_to(q)
    l2.filter_by(t)
    l2.is_in(r)

    l2.scale(4)
    l2.shift(4)
    l2.pad(left=2)
    
    # Test slicing function
    homogeneous_blocks = l2[:2].get_homogeneous_blocks()
    assert homogeneous_blocks[0].block == i.to_rectangle()
    assert homogeneous_blocks[1].block == r

    # Test appending and extending
    assert l + [i2] == Layout([i,q,r,i2])
    assert l + l == Layout([i,q,r]*2)
    l.append(i)
    assert l == Layout([i,q,r,i])
    l2.extend([q])
    assert l2 == Layout([i2, r2, q2, q], page_data={"width": 200, "height": 200})
    
    # Test addition
    l + l2
    with pytest.raises(ValueError):
        l.page_data = {"width": 200, "height":400}
        l + l2

def test_dict():

    i = Interval(1, 2, "y", canvas_height=5)
    i_dict = {
        "block_type": "interval",
        "block_attr": {
            "start": 1,
            "end": 2,
            "axis": "y",
            "canvas_height": 5,
            "canvas_width": 0,
        },
    }
    assert i.to_dict() == i_dict
    assert i == Interval.from_dict(i_dict)

    r = Rectangle(1, 2, 3, 4)
    r_dict = {
        "block_type": "rectangle",
        "block_attr": {"x_1": 1, "y_1": 2, "x_2": 3, "y_2": 4},
    }
    assert r.to_dict() == r_dict
    assert r == Rectangle.from_dict(r_dict)

    q = Quadrilateral(np.arange(8).reshape(4, 2), 200, 400)
    q_dict = {
        "block_type": "quadrilateral",
        "block_attr": {"points": [0, 1, 2, 3, 4, 5, 6, 7], "height": 200, "width": 400},
    }
    assert q.to_dict() == q_dict
    assert q == Quadrilateral.from_dict(q_dict)

    l = Layout([i, r, q], page_data={"width": 200, "height": 200})
    l_dict = {
        "page_data": {"width": 200, "height": 200},
        "blocks": [i_dict, r_dict, q_dict],
    }
    assert l.to_dict() == l_dict

    i2 = TextBlock(i, "")
    i_dict["text"] = ""
    assert i2.to_dict() == i_dict
    assert i2 == TextBlock.from_dict(i_dict)

    r2 = TextBlock(r, id=24)
    r_dict["id"] = 24
    assert r2.to_dict() == r_dict
    assert r2 == TextBlock.from_dict(r_dict)

    q2 = TextBlock(q, text="test", parent=45)
    q_dict["text"] = "test"
    q_dict["parent"] = 45
    assert q2.to_dict() == q_dict
    assert q2 == TextBlock.from_dict(q_dict)

    l2 = Layout([i2, r2, q2])
    l2_dict = {"page_data": {}, "blocks": [i_dict, r_dict, q_dict]}
    assert l2.to_dict() == l2_dict