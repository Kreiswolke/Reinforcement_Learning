# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:12:14 2015

@author: tabea
"""

import numpy as np
import matplotlib.pyplot as plt

class Rat:

    def __init__(self):
        self.maze = False
        self.pickup = False
        self.target = False
        
        # initialize place cell centers
        self.cell_centers = np.zeros((64,2))
        self.cell_centers[0:22,0] = np.arange(2.5,110,5)
        self.cell_centers[0:22,1] = np.ones((22))*57.5
        self.cell_centers[22:44,0] = np.arange(2.5,110,5)
        self.cell_centers[22:44,1] = np.ones((22))*52.5
        self.cell_centers[44:54,0] = np.ones((10))*52.5
        self.cell_centers[44:54,1] = np.arange(2.5,50,5)
        self.cell_centers[54:64,0] = np.ones((10))*57.5        
        self.cell_centers[54:64,1] = np.arange(2.5,50,5)
    
    def in_maze(self,x,y):
        if 50<=x<=60 and 0<=y<50:
            self.maze = True
        elif 0<=x<=110 and 50<=y<=60:
            self.maze = True
        else:
            self.maze = False
    
    def in_pickup(self,x,y):
        if x<90 and self.maze==True:
            self.pickup = False
        else:
            self.pickup = True
            
    def in_target(self,x,y):
        if x<=20 and self.maze==True:
            self.target = True
        else:
            self.target = False
            
    def plot_maze(self):
        plt.plot(self.cell_centers[:,0],self.cell_centers[:,1],'ko')
        plt.xlim(0,110)
        plt.ylim(0,60)
        plt.show()
        


       
