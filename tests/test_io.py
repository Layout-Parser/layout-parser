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

import numpy as np
from layoutparser.elements import Interval, Rectangle, Quadrilateral, TextBlock, Layout
from layoutparser import load_json, load_dict, load_csv, load_pdf

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

    # fmt: off
    assert i == load_dict(i.to_dict()) == load_json("tests/fixtures/io/interval.json")
    assert r == load_dict(r.to_dict()) == load_json("tests/fixtures/io/rectangle.json")
    assert q == load_dict(q.to_dict()) == load_json("tests/fixtures/io/quadrilateral.json")
    assert l == load_dict(l.to_dict()) == load_json("tests/fixtures/io/layout.json")

    assert i2 == load_dict(i2.to_dict()) == load_json("tests/fixtures/io/interval_textblock.json")
    assert r2 == load_dict(r2.to_dict()) == load_json("tests/fixtures/io/rectangle_textblock.json")
    assert q2 == load_dict(q2.to_dict()) == load_json("tests/fixtures/io/quadrilateral_textblock.json")
    assert l2 == load_dict(l2.to_dict()) == load_json("tests/fixtures/io/layout_textblock.json")

    # Test if LP can ignore the unused None features 
    assert l == load_dict(l3.to_dict())
    # fmt: on


def test_csv():
    i = Interval(1, 2, "y", canvas_height=5)
    r = Rectangle(1, 2, 3, 4)
    q = Quadrilateral(np.arange(8).reshape(4, 2), 200, 400)
    l = Layout([i, r, q], page_data={"width": 200, "height": 200})

    _l = load_csv("tests/fixtures/io/layout.csv")
    assert _l != l
    _l.page_data = {"width": 200, "height": 200}
    assert _l == l

    i2 = TextBlock(i, "")
    r2 = TextBlock(r, id=24)
    q2 = TextBlock(q, text="test", parent=45)
    l2 = Layout([i2, r2, q2])

    _l2 = load_csv("tests/fixtures/io/layout_textblock.csv")
    assert _l2 == l2
    

def test_pdf():
    pdf_layout = load_pdf("tests/fixtures/io/example.pdf")
    assert len(pdf_layout) == 1
    
    page_layout = pdf_layout[0]
    for attr_name in ["width", "height", "index"]:
        assert attr_name in page_layout.page_data

    assert len(set(ele.type for ele in page_layout)) == 3
    # Only three types of font show-up in the file
    
def test_empty_pdf():
    pdf_layout = load_pdf("tests/fixtures/io/empty.pdf")
    assert len(pdf_layout) == 0