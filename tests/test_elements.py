from layoutparser.elements import Interval, Rectangle, Quadrilateral, TextBlock, Layout
import numpy as np
import pandas as pd

def test_interval():
    
    i = Interval(1, 2, axis='y', canvas_height=30, canvas_width=400)
    i.to_rectangle()
    i.to_quadrilateral()
    assert i.shift(1) == Interval(2, 3, axis='y', canvas_height=30, canvas_width=400)
    
    i = Interval(1, 2, axis='x')
    assert i.shift([1,2]) == Interval(2, 3, axis='x')
    assert i.scale([2,1]) == Interval(2, 4, axis='x')
    assert i.pad(left=10, right=20) == Interval(0, 22)  # Test the safe_mode
    assert i.pad(left=10, right=20, safe_mode=False) == Interval(-9, 22) 

    img = np.random.randint(12, 24, (40,40))
    img[:, 10:20] = 0
    i = Interval(5, 11, axis='x')
    assert np.unique(i.crop_image(img)[:, -1]) == np.array([0])
    
def test_rectangle():
    
    r = Rectangle(1, 2, 3, 4)
    r.to_interval()
    r.to_quadrilateral()
    assert r.pad(left=1, right=5, top=2, bottom=4) == Rectangle(0, 0, 8, 8)
    assert r.shift([1,2]) == Rectangle(2, 4, 4, 6)
    assert r.shift(1) == Rectangle(2, 3, 4, 5)
    assert r.scale([3,2]) == Rectangle(3, 4, 9, 8)
    assert r.scale(2) == Rectangle(2, 4, 6, 8)
    
    img = np.random.randint(12, 24, (40,40))
    r.crop_image(img).shape == (2, 2)
    
def test_quadrilateral():
    
    points = np.array([[2, 2], [6, 2], [6,7], [2,6]])
    q = Quadrilateral(points)
    q.to_interval()
    q.to_rectangle()
    assert q.shift(1) == Quadrilateral(points + 1)
    assert q.shift([1,2]) == Quadrilateral(points + np.array([1,2]))
    assert q.scale(2) == Quadrilateral(points * 2)
    assert q.scale([3,2]) == Quadrilateral(points * np.array([3,2]))
    assert q.pad(left=1, top=2, bottom=4) == Quadrilateral(np.array([[1, 0], [6, 0], [6, 11], [1, 10]]))
    assert (q.mapped_rectangle_points == np.array([[0,0],[4,0],[4,5],[0,5]])).all()

    points = np.array([[2, 2], [6, 2], [6,5], [2,5]])
    q = Quadrilateral(points)
    img = np.random.randint(2, 24, (30, 20)).astype('uint8')
    img[2:5, 2:6] = 0
    assert np.unique(q.crop_image(img)) == np.array([0])
    
def test_interval_relations():
    
    i = Interval(4, 5, axis='y')
    r = Rectangle(3, 3, 5, 6)
    q = Quadrilateral(np.array([[2,2],[6,2],[6,7],[2,5]]))
    
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
    
    i = Interval(4, 5, axis='y')
    q = Quadrilateral(np.array([[2,2],[6,2],[6,7],[2,5]]))
    r = Rectangle(3, 3, 5, 6)
    
    assert not r.is_in(q)
    assert r.is_in(q, soft_margin={"bottom":1})
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
    
    i = Interval(4, 5, axis='y')
    q = Quadrilateral(np.array([[2,2],[6,2],[6,7],[2,5]]))
    r = Rectangle(3, 3, 5, 6)
    
    assert not q.is_in(r)
    assert q.is_in(i, soft_margin={"top":2, "bottom":2})
    assert q.is_in(r, soft_margin={"left":1, "top":1, "right":1,"bottom":1})
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
        
    i = Interval(4, 5, axis='y')
    q = Quadrilateral(np.array([[2,2],[6,2],[6,7],[2,5]]))
    r = Rectangle(3, 3, 5, 6)
    
    t = TextBlock(i, id=1, type=2, text="12")
    assert t.relative_to(q).condition_on(q).block == i.put_on_canvas(q).to_quadrilateral()
    t = TextBlock(r, id=1, type=2, parent="a")
    assert t.relative_to(i).condition_on(i).block == r
    t = TextBlock(q, id=1, type=2, parent="a")
    assert t.relative_to(r).condition_on(r).block == q
    
    # Ensure the operations did not change the object itself
    assert t == TextBlock(q, id=1, type=2, parent="a")
    t1 = TextBlock(q, id=1, type=2, parent="a")
    t2 = TextBlock(i, id=1, type=2, text="12")
    t1.relative_to(t2)
    assert t2.is_in(t1)
    
    t = TextBlock(q, score=0.2)

def test_layout():
    i = Interval(4, 5, axis='y')
    q = Quadrilateral(np.array([[2,2],[6,2],[6,7],[2,5]]))
    r = Rectangle(3, 3, 5, 6)
    t = TextBlock(i, id=1, type=2, text="12")
    
    l = Layout([i, q, r])
    l.get_texts()
    l.condition_on(i)
    l.relative_to(q)
    l.filter_by(t)
    l.is_in(r)
    
    l = Layout([
        TextBlock(i, id=1, type=2, text="12"),
        TextBlock(r, id=1, type=2, parent="a"),
        TextBlock(q, id=1, type=2, next="a")
    ])
    l.get_texts()
    l.get_info('next')
    l.condition_on(i)
    l.relative_to(q)
    l.filter_by(t)
    l.is_in(r)
    
    l.scale(4)
    l.shift(4)
    l.pad(left=2)
    
def test_df():
    
    df = pd.DataFrame(
        columns=\
            ["_identifier", "x_1", "y_1", "x_2", "y_2", "height", "width", "p11", "p12", "p21", "p22", "p31", "p32", "p41", "p42"],
        data=[
            ['_interval',   None,  10,    None,  12,     240,      None,   None,  None,  None,  None,  None,  None,  None,  None ],
            ['_interval',   12,    None,  24,    None,   120,      50,     None,  None,  None,  None,  None,  None,  None,  None ],
            ['_interval',   0,     10,    0,     12,     120,      50,     None,  None,  None,  None,  None,  None,  None,  None ], # for fillna with 0
            ['_rectangle',  12,    32,    24,    55,     None,     None,   None,  None,  None,  None,  None,  None,  None,  None ],
            ['_rectangle',  12,    32,    24,    55,     0,        0,      None,  None,  None,  None,  None,  None,  None,  None ],
            ['_quadrilateral',None,None,  None,  None,   None,     None,   1,     2,     3,     2,     3,     6,     1,     4    ],
            ['_quadrilateral',None,None,  None,  None,   0,        0,      1,     2,     3,     2,     3,     6,     1,     4    ],
            ['_quadrilateral',0,   0,     0,     0,      0,        0,      1,     2,     3,     2,     3,     6,     1,     4    ],
        ]
    )
    
    layout = Layout.from_dataframe(df)
    assert layout[0] == Interval(10, 12, 'y', canvas_height=240)
    assert layout[2] == Interval(10, 12, 'y', canvas_height=120, canvas_width=50)
    
    assert layout[3] == Rectangle(x_1=12, y_1=32, x_2=24, y_2=55)
    assert layout[3] == layout[4]

    assert not layout[5] == Quadrilateral(np.arange(8).reshape(4,-1))
    assert layout[6] == Quadrilateral(np.array([[1,2], [3,2], [3,6], [1,4]]))
    
    df = pd.DataFrame(
        columns=\
            ["_identifier", "x_1", "y_1", "x_2", "y_2", "height", "width", "p11", "p12", "p21", "p22", "p31", "p32", "p41", "p42", 'next', 'parent'],
        data=[
            ['_interval',   None,  10,    None,  12,     240,      None,   None,  None,  None,  None,  None,  None,  None,  None,  None,   None    ],
            ['_interval',   12,    None,  24,    None,   120,      50,     None,  None,  None,  None,  None,  None,  None,  None,  None,   None    ],
            ['_interval',   0,     10,    0,     12,     120,      50,     None,  None,  None,  None,  None,  None,  None,  None,  None,   24      ], 
                # for fillna with 0
            ['_rectangle',  12,    32,    24,    55,     None,     None,   None,  None,  None,  None,  None,  None,  None,  None,  None,   None    ],
            ['_rectangle',  12,    32,    24,    55,     0,        0,      None,  None,  None,  None,  None,  None,  None,  None,  12,     None    ],
            ['_quadrilateral',None,None,  None,  None,   None,     None,   1,     2,     3,     2,     3,     6,     1,     4,     None,   None    ],
            ['_quadrilateral',None,None,  None,  None,   0,        0,      1,     2,     3,     2,     3,     6,     1,     4,     None,   None    ],
            ['_textblock',  None,None,  None,  None,     0,        0,      1,     2,     3,     2,     3,     6,     1,     4,     None,   28      ],
        ]
    )
    
    layout = Layout.from_dataframe(df)
    assert layout[0] == Interval(10, 12, 'y', canvas_height=240)
    assert layout[2] == Interval(10, 12, 'y', canvas_height=120, canvas_width=50)
    
    assert layout[3] == Rectangle(x_1=12, y_1=32, x_2=24, y_2=55)
    assert layout[3] == layout[4]
    assert layout[6] == Quadrilateral(np.array([[1,2], [3,2], [3,6], [1,4]]))
    
    assert layout[-1].block == Quadrilateral(np.array([[1,2], [3,2], [3,6], [1,4]]))
    assert layout[-1].parent == 28
    
    
    df = pd.DataFrame(
        columns=\
            ["x_1", "y_1", "x_2", "y_2", "height", "width", "p11", "p12", "p21", "p22", "p31", "p32", "p41", "p42", 'next', 'parent'],
        data=[
            [None,  10,    None,  12,     240,      None,   None,  None,  None,  None,  None,  None,  None,  None,  None,   None    ],
            [12,    None,  24,    None,   120,      50,     None,  None,  None,  None,  None,  None,  None,  None,  None,   None    ],
            [0,     10,    0,     12,     120,      50,     None,  None,  None,  None,  None,  None,  None,  None,  None,   24      ], 
            # for fillna with 0
            [12,    32,    24,    55,     None,     None,   None,  None,  None,  None,  None,  None,  None,  None,  None,   None    ],
            [12,    32,    24,    55,     None,     None,   None,  None,  None,  None,  None,  None,  None,  None,  12,     None    ],
            [None,  None,  None,  None,   None,     None,   1,     2,     3,     2,     3,     6,     1,     4,     None,   None    ],
            [None,  None,  None,  None,   0,        0,      1,     2,     3,     2,     3,     6,     1,     4,     None,   None    ],
            [None,  None,  None,  None,   0,        0,      1,     2,     3,     2,     3,     6,     1,     4,     None,   28      ],
        ]
    )
    
    layout = Layout.from_dataframe(df)
    assert layout[0].block == Interval(10, 12, 'y', canvas_height=240)
    assert layout[2].block == Interval(10, 12, 'y', canvas_height=120, canvas_width=50)
    
    assert layout[3].block == Rectangle(x_1=12, y_1=32, x_2=24, y_2=55)
    assert not layout[3] == layout[4]
    assert layout[6].block == Quadrilateral(np.array([[1,2], [3,2], [3,6], [1,4]]))
    
    assert layout[-1].block == Quadrilateral(np.array([[1,2], [3,2], [3,6], [1,4]]))
    assert layout[-1].parent == 28
    
    df = pd.DataFrame(
        columns=\
            ["x_1", "y_1", "x_2", "y_2"],
        data=[
            [0,     10,    0,     12,  ],
            [12,    32,    24,    55,  ],
        ])
    
    layout = Layout.from_dataframe(df)
    assert layout[1] == Rectangle(x_1=12, y_1=32, x_2=24, y_2=55)
    
    
    df = pd.DataFrame(
        columns=\
            ["x_1", "y_1", "x_2", "y_2", "height", "width"],
        data=[
            [0,     10,    0,     12,    240,      520    ],
            [12,    None,  24,    None,  240,      None   ],
        ])
    
    layout = Layout.from_dataframe(df)
    assert layout[1] == Interval(12, 24, 'x', canvas_height=240)
    
    
    df = pd.DataFrame(
        columns=\
            ["p11", "p12", "p21", "p22", "p31", "p32", "p41", "p42", 'width', 'height'],
        data=[
            [1,     2,     3,     2,     3,     6,     1,     4,     None,   None    ],
            [1,     2,     3,     2,     3,     6,     1,     4,     None,   None    ],
            [1,     2,     3,     2,     3,     6,     1,     4,     None,   28      ],
        ])
    
    layout = Layout.from_dataframe(df)
    assert layout[1] == Quadrilateral(np.array([[1,2], [3,2], [3,6], [1,4]]))
    assert layout[2] == Quadrilateral(np.array([[1,2], [3,2], [3,6], [1,4]]), height=28)