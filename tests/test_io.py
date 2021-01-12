import numpy as np
from layoutparser.elements import Interval, Rectangle, Quadrilateral, TextBlock, Layout
from layoutparser.io import load_json, load_dict

def test_json():
    
    i = Interval(1, 2, "y", canvas_height=5)
    r = Rectangle(1, 2, 3, 4)
    q = Quadrilateral(np.arange(8).reshape(4, 2), 200, 400)
    l = Layout([i, r, q], page_data={"width": 200, "height": 200})

    i2 = TextBlock(i, "")
    r2 = TextBlock(r, id=24)
    q2 = TextBlock(q, text="test", parent=45)
    l2 = Layout([i2, r2, q2])

    i3 = TextBlock(i, None)
    r3 = TextBlock(r, id=None)
    q3 = TextBlock(q, text=None, parent=None)
    l3 = Layout([i3, r3, q3], page_data={"width": 200, "height": 200})
    
    assert i == load_dict(i.to_dict()) == load_json("tests/fixtures/io/interval.json")
    assert r == load_dict(r.to_dict()) == load_json("tests/fixtures/io/rectangle.json")
    assert q == load_dict(q.to_dict()) == load_json("tests/fixtures/io/quadrilateral.json")
    assert l == load_dict(l.to_dict()) == load_json("tests/fixtures/io/layout.json")

    assert i2 == load_dict(i2.to_dict()) == load_json("tests/fixtures/io/interval_textblock.json")
    assert r2 == load_dict(r2.to_dict()) == load_json("tests/fixtures/io/rectangle_textblock.json")
    assert q2 == load_dict(q2.to_dict()) == load_json("tests/fixtures/io/quadrilateral_textblock.json")
    assert l2 == load_dict(l2.to_dict()) == load_json("tests/fixtures/io/layout_textblock.json")

    assert l == load_dict(l3.to_dict())