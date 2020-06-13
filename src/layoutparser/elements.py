from abc import ABC, abstractmethod
import numpy as _np

def _cvt_coordinates_to_points(coords):

    x_1, y_1, x_2, y_2 = coords
    return _np.array([[x_1, y_1], # Top Left
                   [x_2, y_1], # Top Right
                   [x_2, y_2], # Bottom Right
                   [x_1, y_2], # Bottom Left
                  ])

class BaseLayoutElement(ABC):

    #######################################################################
    #########################  Layout Properties  #########################
    #######################################################################
    
    @property
    @abstractmethod
    def width(self): pass
    
    @property
    @abstractmethod
    def height(self): pass
    
    @property
    @abstractmethod
    def coordinates(self): pass
    
    @property
    @abstractmethod
    def points(self): pass
    
    #######################################################################
    ### Geometric Relations (relative to, condition on, and is in)  ### 
    #######################################################################
    
    @abstractmethod
    def condition_on(self): pass
    
    @abstractmethod
    def relative_to(self): pass
    
    @abstractmethod
    def is_in(self): pass
    
    #######################################################################
    ############### Geometric Operations (pad, shift, scale) ##############
    #######################################################################
    
    @abstractmethod
    def pad(self): pass
    
    @abstractmethod
    def shift(self): pass
    
    @abstractmethod
    def scale(self): pass
    
    #######################################################################
    ################################# MISC ################################
    #######################################################################
    
    @abstractmethod
    def crop_image(self): pass
    
    def __repr__(self):

        info_str = ', '.join([f'{key}={val}' for key, val in vars(self).items()])
        return f"{self.__class__.__name__}({info_str})"

    def __eq__(self, other):
        
        if other.__class__ is not self.__class__:
            return False
        
        return vars(self) == vars(other)


class Interval(BaseLayoutElement):

    def __init__(self, start, end, axis='x',
                    img_height=0, img_width=0):
        
        assert start<=end, f"Invalid input for start and end. Start must <= end."
        self.start = start
        self.end   = end
        
        assert axis in ['x', 'y'], f"Invalid axis {axis}. Axis must be in 'x' or 'y'"
        self.axis  = axis
        
        self.img_height = img_height
        self.img_width  = img_width
    
    @property
    def height(self):
        if self.axis == 'x': return self.img_height
        else: return self.end - self.start
        
    @property
    def width(self):
        if self.axis == 'y': return self.img_width
        else: return self.end - self.start

    @property
    def coordinates(self):
        if self.axis == 'x':
            coords = (self.start, 0, self.end, self.img_height)
        else:
            coords = (0, self.start, self.img_width, self.end)
        
        return coords
    
    @property
    def points(self):
        return _cvt_coordinates_to_points(self.coordinates)
    
    def condition_on(self, other): pass
    
    def relative_to(self, other): pass
    
    def is_in(self, other): pass
    
    def pad(self): pass
    
    def shift(self): pass
    
    def scale(self): pass
    
    def crop_image(self): pass
    
    def to_rectangle(self): 
        return Rectangle(*self.coordinates)
    
    def to_quadrilateral(self):
        return Quadrilateral(self.points)
    

class Rectangle(BaseLayoutElement):
    
    def __init__(self, x_1, y_1, x_2, y_2):
        
        self.x_1 = x_1
        self.y_1 = y_1
        self.x_2 = x_2
        self.y_2 = y_2
        
    @property
    def height(self):
        return self.y_2 - self.y_1
    
    @property
    def width(self):
        return self.x_2 - self.x_1
     
    @property
    def coordinates(self):
        return (self.x_1, self.y_1, self.x_2, self.y_2)
    
    @property
    def points(self):
        return _cvt_coordinates_to_points(self.coordinates)
    
    @property
    def center(self):
        return (self.x_1 + self.x_2)/2., (self.y_1 + self.y_2)/2.
    
    def condition_on(self, other): pass
    
    def relative_to(self, other): pass
    
    def is_in(self, other): pass
    
    def pad(self): pass
    
    def shift(self): pass
    
    def scale(self): pass
    
    def crop_image(self): pass
    
    def to_interval(self, axis='x', **kwargs):
        if axis == 'x':
            start, end = self.x_1, self.x_2
        else:
            start, end = self.y_1, self.y_2
            
        return Interval(start, end, axis=axis, **kwargs)
    
    def to_quadrilateral(self):
        return Quadrilateral(self.points)

class Quadrilateral(BaseLayoutElement):
    
    def __init__(self, points, width=None, height=None):
        
        assert isinstance(points, _np.ndarray), f" Invalid input: points must be a numpy array"
        
        self._points = points
        self._width  = width
        self._height = height
        
    @property
    def height(self):
        return self.points[:,1].max() - self.points[:,1].min()
    
    @property
    def width(self):
        return self.points[:,0].max() - self.points[:,0].min()
     
    @property
    def coordinates(self):
        x_1 = self.points[:,0].min()
        y_1 = self.points[:,1].min()
        x_2 = self.points[:,0].max()
        y_2 = self.points[:,1].max()
        return (x_1, y_1, x_2, y_2)
    
    @property
    def points(self):
        return self._points
    
    @property
    def center(self):
        return tuple(self.points.mean(axis=0).tolist())
    
    def condition_on(self, other): pass
    
    def relative_to(self, other): pass
    
    def is_in(self, other): pass
    
    def pad(self): pass
    
    def shift(self): pass
    
    def scale(self): pass
    
    def crop_image(self): pass
    
    def to_interval(self, axis='x', **kwargs):

        x_1, y_1, x_2, y_2 = self.coordinates
        if axis == 'x':
            start, end = x_1, x_2
        else:
            start, end = y_1, y_2
            
        return Interval(start, end, axis=axis, **kwargs)
    
    def to_rectangle(self):
        return Rectangle(*self.coordinates)