from abc import ABC, abstractmethod


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
