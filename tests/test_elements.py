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
    
def test_quadrilateral():
    
    q = Quadrilateral(np.array([[1, 2], [3, 4], [5,6], [7,8]]))
    q.to_interval()
    q.to_rectangle()