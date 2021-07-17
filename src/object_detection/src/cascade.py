import os
import time
import math
import copy

import cv2
import numpy as np

class State:
    ''' 
        State class maintains a state of points of interest in a camera feed. 

        Variables: 
            self.poi - Nx5 list of Points Of Interest. The 5 values are: x, y, x+w, h+h, and weight
            self.min_weight - threshold of what weight must be surpassed to be considered a valuable point
            self.max_distance - how far apart two detection can be in coincident frames and still be considered
                a read on the same object
            self.decay - factor by which weights decay with each new frame 

        Functions:
            draw(img) - given an image (of same size of the video frames), return the image with
                bounding boxes drawn around each point of interest that has a larger weight than self.min_weight
            update(data) - takes Nx4 list of rectangles (x, y, x+w, y+h) that are identified from a haar cascade model
                and adjusts weights/adds points of interest
            run_cascades(img) - runs all model cascade models. Use this function with each frame wanting to be processed
            get_pois() - returns a Nx4 list of all points of interest above the min_weight. Feed this data into any
                external classifiers. 
    
    '''
    def __init__(self):
        self.poi = np.empty([1,5]) # points of interest
        self.vor = None
        self.min_weight = 0.3
        self.max_distance = 200
        self.decay = 0.8

        fileDir = os.path.dirname(os.path.realpath('__file__')) # __file__ = main.py 
        self.ss_cascade = cv2.CascadeClassifier(os.path.join(fileDir, '.\models\stopsigns.xml'))
 
    def draw(self, img):
        ret = copy.copy(img)
        if len(self.poi) > 1:
            index = np.argmax(self.poi[:,4])
            if index > 0:
                [x, y, w, h, s] = self.poi[index]
                color = (255, 0, 0) if len(ret.shape) == 3 else 255
                cv2.rectangle(ret, (int(x), int(y)), (int(x + w), int(y + h)), color, math.ceil(s))
        return ret 

    def update(self, data):
        '''
        Alogirthm:
            1) check closest point of same type incoming
            2) increase both points by 1 if so 
            3) reduce weight and filter of all points 
        '''
        # Add new detections, increase weights if 
        for d in data:
            for (x, y, w, h) in d:    
                if len(self.poi) > 0:
                    closest_index = np.argmin(np.sum((self.poi[:,0:2] - np.array([x,y]))**2, axis=1))

                    if np.sum((self.poi[closest_index,0:4] - np.array([x,y,h,w]))**2) < self.max_distance:
                        self.poi = np.vstack([self.poi,np.array([x, y, w, h, 1 + self.poi[closest_index,4]])])
                        self.poi[closest_index,4] *= 0.5
                    else:
                        self.poi = np.vstack([self.poi,np.array([x, y, w, h, 1])])
                else:
                    self.poi = np.vstack([self.poi,np.array([x, y, w, h, 1])])

        # Decrease the weights of each detection 
        i = 1
        while i < len(self.poi):
            self.poi[i,4] *= self.decay
            if self.poi[i,4] < self.min_weight:
                self.poi = np.delete(self.poi, i, axis=0)
            else:
                i += 1

    def run_cascades(self, img):  
        ss = self.ss_cascade.detectMultiScale(img, 2, 2)
        self.update([ss])
        
        return ss, [], [], []

    def get_pois(self):
        ret = []
        for i in range(0, len(self.poi)):
            if self.poi[i,4] > self.min_weight:
                ret += self.poi[i,:4]
        return np.array(ret)
