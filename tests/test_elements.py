from layoutparser.elements import Interval, Rectangle, Quadrilateral
import numpy as np

def test_interval():
    
    i = Interval(1, 2, axis='y', img_height=30, img_width=400)
    i.to_rectangle()
    i.to_quadrilateral()
    assert i.shift(1) == Interval(2, 3, axis='y', img_height=30, img_width=400)
    
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