from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import copy
import warnings
import numpy as _np
from cv2 import getPerspectiveTransform as _getPerspectiveTransform
from cv2 import warpPerspective as _warpPerspective

def _cvt_coordinates_to_points(coords):

    x_1, y_1, x_2, y_2 = coords
    return _np.array([[x_1, y_1], # Top Left
                   [x_2, y_1], # Top Right
                   [x_2, y_2], # Bottom Right
                   [x_1, y_2], # Bottom Left
                  ])

def _cvt_points_to_coordinates(points):
    x_1 = points[:,0].min()
    y_1 = points[:,1].min()
    x_2 = points[:,0].max()
    y_2 = points[:,1].max()
    return (x_1, y_1, x_2, y_2)

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
    
    def set(self, inplace=False, **kwargs):
        
        obj = self if inplace else copy(self)
        var_dict = vars(obj)
        for key, val in kwargs.items():
            if key in var_dict:
                var_dict[key] = val
            elif f"_{key}" in var_dict:
                var_dict[f"_{key}"] = val
            else:
                raise ValueError(f"Unkonwn attribute name: {key}")

        return obj
    
    def __repr__(self):

        info_str = ', '.join([f'{key}={val}' for key, val in vars(self).items()])
        return f"{self.__class__.__name__}({info_str})"

    def __eq__(self, other):
        
        if other.__class__ is not self.__class__:
            return False
        
        return vars(self) == vars(other)


class Interval(BaseLayoutElement):

    def __init__(self, start, end, axis='x',
                    canvas_height=0, canvas_width=0):
        
        assert start<=end, f"Invalid input for start and end. Start must <= end."
        self.start = start
        self.end   = end
        
        assert axis in ['x', 'y'], f"Invalid axis {axis}. Axis must be in 'x' or 'y'"
        self.axis  = axis
        
        self.canvas_height = canvas_height
        self.canvas_width  = canvas_width
    
    @property
    def height(self):
        if self.axis == 'x': return self.canvas_height
        else: return self.end - self.start
        
    @property
    def width(self):
        if self.axis == 'y': return self.canvas_width
        else: return self.end - self.start

    @property
    def coordinates(self):
        if self.axis == 'x':
            coords = (self.start, 0, self.end, self.canvas_height)
        else:
            coords = (0, self.start, self.canvas_width, self.end)
        
        return coords
    
    @property
    def points(self):
        return _cvt_coordinates_to_points(self.coordinates)
    
    @property
    def center(self): 
        return (self.start + self.end) / 2.

    def put_on_canvas(self, canvas):
        if isinstance(canvas, _np.ndarray):
            h, w = canvas.shape[:2]
        elif isinstance(canvas, BaseLayoutElement):
            h, w = canvas.height, canvas.width
        else:
            raise NotImplementedError
        
        return self.set(canvas_height=h, canvas_width=w)

    def relative_to(self, other): pass
    
    def is_in(self, other): pass
    
    def pad(self, left=0, right=0, top=0, bottom=0, safe_mode=True):
        
        if self.axis == 'x':
            start = self.start - left
            end   = self.end   + right
            if top or bottom:
                warnings.warn(f"Invalid padding top/bottom for an x axis {self.__class__.__name__}")
        else:
            start = self.start - top
            end   = self.end   + bottom
            if left or right:
                warnings.warn(f"Invalid padding right/left for a y axis {self.__class__.__name__}")
                
        if safe_mode:
            start = max(0, start)
        
        return self.set(start=start, end=end)

    def shift(self, shift_distance): 
        
        if isinstance(shift_distance, Iterable):        
            shift_distance = shift_distance[0] if self.axis == 'x' \
                                else shift_distance[1]
            warnings.warn(f"Input shift for multiple axes. Only use the distance for the {self.axis} axis")

        start = self.start + shift_distance
        end   = self.end   + shift_distance
        return self.set(start=start, end=end)
        
    def scale(self, scale_factor): 
        
        if isinstance(scale_factor, Iterable):        
            scale_factor = scale_factor[0] if self.axis == 'x' \
                                else scale_factor[1]
            warnings.warn(f"Input scale for multiple axes. Only use the factor for the {self.axis} axis")
            
        start = self.start * scale_factor
        end   = self.end   * scale_factor
        return self.set(start=start, end=end)

    def crop_image(self, image): 
        x_1, y_1, x_2, y_2 = self.put_on_canvas(image).coordinates
        return image[int(y_1):int(y_2), int(x_1):int(x_2)]
    
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
    
    def pad(self, left=0, right=0, top=0, bottom=0, 
                safe_mode=True):
        
        x_1 = self.x_1 - left
        y_1 = self.y_1 - top
        x_2 = self.x_2 + right
        y_2 = self.y_2 + bottom
        
        if safe_mode:
            x_1 = max(0, x_1)
            y_1 = max(0, y_1)
        
        return self.__class__(x_1, y_1, x_2, y_2)
    
    def shift(self, shift_distance=0): 
        
        if not isinstance(shift_distance, Iterable):
            shift_x = shift_distance
            shift_y = shift_distance
        else:
            assert len(shift_distance) == 2, "scale_factor should have 2 elements, one for x dimension and one for y dimension"
            shift_x, shift_y = shift_distance
            
        x_1 = self.x_1 + shift_x
        y_1 = self.y_1 + shift_y
        x_2 = self.x_2 + shift_x
        y_2 = self.y_2 + shift_y
        return self.__class__(x_1, y_1, x_2, y_2)
    
    def scale(self, scale_factor=1):
        
        if not isinstance(scale_factor, Iterable):
            scale_x = scale_factor
            scale_y = scale_factor
        else:
            assert len(scale_factor) == 2, "scale_factor should have 2 elements, one for x dimension and one for y dimension"
            scale_x, scale_y = scale_factor
        
        x_1 = self.x_1 * scale_x
        y_1 = self.y_1 * scale_y
        x_2 = self.x_2 * scale_x
        y_2 = self.y_2 * scale_y
        return self.__class__(x_1, y_1, x_2, y_2)

    def crop_image(self, image): 
        x_1, y_1, x_2, y_2 = self.coordinates
        return image[int(y_1):int(y_2), int(x_1):int(x_2)]
    
    def to_interval(self, axis='x', **kwargs):
        if axis == 'x':
            start, end = self.x_1, self.x_2
        else:
            start, end = self.y_1, self.y_2
            
        return Interval(start, end, axis=axis, **kwargs)
    
    def to_quadrilateral(self):
        return Quadrilateral(self.points)


class Quadrilateral(BaseLayoutElement):
    
    def __init__(self, points, height=None, width=None):
        
        assert isinstance(points, _np.ndarray), f" Invalid input: points must be a numpy array"
        
        self._points = points
        self._width  = width
        self._height = height
        
    @property
    def height(self):
        if self._height is not None:
            return self._height
        return self.points[:,1].max() - self.points[:,1].min()
    
    @property
    def width(self):
        if self._width is not None:
            return self._width
        return self.points[:,0].max() - self.points[:,0].min()
     
    @property
    def coordinates(self):
        return _cvt_points_to_coordinates(self.points)
    
    @property
    def points(self):
        return self._points
    
    @property
    def center(self):
        return tuple(self.points.mean(axis=0).tolist())
    
    @property
    def mapped_rectangle_points(self):

        x_map = {0:0, 1:0, 2:self.width, 3:self.width}
        y_map = {0:0, 1:0, 2:self.height, 3:self.height}

        return self.map_to_points_ordering(x_map, y_map)
    
    @property
    def perspective_matrix(self):
        return _getPerspectiveTransform(self.points.astype('float32'),
                                        self.mapped_rectangle_points.astype('float32'))
    
    def map_to_points_ordering(self, x_map, y_map):
        
        points_ordering = self.points.argsort(axis=0).argsort(axis=0)
        # Ref: https://github.com/numpy/numpy/issues/8757#issuecomment-355126992
        
        return _np.vstack([
                    _np.vectorize(x_map.get)(points_ordering[:,0]),
                    _np.vectorize(y_map.get)(points_ordering[:,1])
                ]).T
    
    def condition_on(self, other): pass
    
    def relative_to(self, other): pass
    
    def is_in(self, other): pass
    
    def pad(self, left=0, right=0, top=0, bottom=0, 
                safe_mode=True):
        
        x_map = {0:-left,  1:-left,  2:right,  3:right}
        y_map = {0:-top,   1:-top,   2:bottom, 3:bottom}
        
        padding_mat  = self.map_to_points_ordering(x_map, y_map)
        
        points = self.points + padding_mat
        if safe_mode:
            points = _np.maximum(points, 0)
        
        return self.set(points=points)

    def shift(self, shift_distance=0):
        
        if not isinstance(shift_distance, Iterable):
            shift_mat = [shift_distance, shift_distance]
        else:
            assert len(shift_distance) == 2, "scale_factor should have 2 elements, one for x dimension and one for y dimension"
            shift_mat = shift_distance
        
        points = self.points + _np.array(shift_mat)
        
        return self.set(points=points)
        
    def scale(self, scale_factor=1):
        
        if not isinstance(scale_factor, Iterable):
            scale_mat = [scale_factor, scale_factor]
        else:
            assert len(scale_factor) == 2, "scale_factor should have 2 elements, one for x dimension and one for y dimension"
            scale_mat = scale_factor
        
        points = self.points * _np.array(scale_mat)    

        return self.set(points=points)

    def crop_image(self, image):
        return _warpPerspective(image, self.perspective_matrix, (int(self.width), int(self.height)))
    
    def to_interval(self, axis='x', **kwargs):

        x_1, y_1, x_2, y_2 = self.coordinates
        if axis == 'x':
            start, end = x_1, x_2
        else:
            start, end = y_1, y_2
            
        return Interval(start, end, axis=axis, **kwargs)
    
    def to_rectangle(self):
        return Rectangle(*self.coordinates)
    
    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return False
        return _np.isclose(self.points, other.points).all()
    
    def __repr__(self):
        keys = ['points', 'width', 'height']
        info_str = ', '.join([f'{key}={getattr(self, key)}' for key in keys])
        return f"{self.__class__.__name__}({info_str})"