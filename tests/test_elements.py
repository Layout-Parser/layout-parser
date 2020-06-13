from layoutparser.elements import Interval, Rectangle, Quadrilateral
import numpy as np

def test_interval():
    
    Interval(1, 2, axis='x')
    Interval(1, 2, axis='y')
    i = Interval(1, 2, axis='y', img_height=30, img_width=400)
    i.to_rectangle()
    i.to_quadrilateral()
    
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
    
    q = Quadrilateral(np.array([[1, 2], [3, 4], [5,6], [7,8]]))
    q.to_interval()
    q.to_rectangle()