import copy

import cv2
import numpy as np

from . import image_tools
from . import cascade

class Frame_Handler:
    def __init__(self):
        self.state = cascade.State() # Contains state and cascade data
        self.next_frame = np.array([]) # Next frame to be processed, live with camera
        self.current_frame = np.array([]) # Current frame being processed 
        self.drawn_frame = np.array([]) # Bounding box overlay for visualization 

        self.ss = np.array([]) # Cropped images of stop signs 
        self.arrows = np.array([]) # """ arrows
        self.lights = np.array([]) # """ lights
        self.cones = np.array([]) # """ cones 

    def process_frame(self):
        ''' Frame handling function. Call this at operating rate in loop '''
        self.current_frame = copy.copy(self.next_frame)
        ss, arrows, lights, cones = self.state.run_cascades(self.current_frame)
        self.drawn_frame = self.state.draw(self.current_frame)

        self.ss, self.ss_loc = self.process_bbox(ss)
        self.arrows, self.arrows_loc = self.process_bbox(arrows)
        self.lights, self.lights_loc = self.process_bbox(lights)
        self.cones, self.cones_loc = self.process_bbox(cones)

    def process_bbox(self, bbox):
        return_images = []
        return_loc = [] # TO BE COMPLETED
        for box in bbox:
            to_add = self.current_frame[box[1]:box[1]+box[2], box[0]:box[0]+box[2]]
            return_images += [np.array(to_add)]
        
        return np.array(return_images), return_loc

    def update_from_camera(self, img):
        ''' Camera callback function '''
        self.next_frame = image_tools.scale(img, 800)
    