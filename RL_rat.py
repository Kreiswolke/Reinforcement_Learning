# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:12:14 2015

@author: tabea
"""

class Rat:

    def _init_():
        self.maze = False
        self.pickup = False
        self.target = False
        self.cell_arr = np.zeros((64,2))
    
    def in_maze(x,y):
        if 50<=x<=60 and 0<=y<50:
            self.maze = True
        elif 0<=x<=110 and 50<=y<=60:
            self.maze = True
        else:
            self.maze = False
    
    def in_pickup(x,y):
        if x<90 and self.maze==True:
            self.pickup = False
        else:
            self.pickup = True
            
    def in_target(x,y):
        if x<=20 and self.maze==True:
            self.target = True
        else:
            self.target = False
            
    def plot_maze():
        
        
    def set_place_cells():
        