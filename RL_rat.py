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
        
        self.rates = np.zeros(self.cell_centers.shape[0])
        self.beta = 0
        self.rat_position = np.zeros(2)
    
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
        plt.plot(self.cell_centers[:,0],self.cell_centers[:,1],'ko',markersize=0.8)
        ax = plt.gca()
        ax.axhspan(50,60,alpha=0.15,color='gray')
        ax.axvspan(50,60,ymax=5/6,alpha=0.15,color='gray')
        ax.axis('off')
        plt.xlim(0,110)
        plt.ylim(0,60)
        plt.show()
        
    def in_target(self,x,y):
        if x<=20 and self.maze==True:
            self.target = True
        else:
            self.target = False
                
        
    def get_firing_rate(self, sigma = 5):
        if self.beta == self.pickup:
            #np.exp(-((self.cell_centers[0,:]- self.position[0])**2 \
            #+ self.cell_centers[1,:]- self.position[1])**2)/(2*sigma**2))
            for j in range(self.cell_centers.shape[0]):
                self.rates[j] = np.exp(-((np.linalg.norm(self.rat_position - \
                self.cell_centers[j,:])) ** 2) / (2 * (sigma ** 2)))

        else:
            self.rates = np.zeros_like(self.rates)
            
    def get_velocity(self):
        return numpy.random.normal(3, 1.5)
        
    def choose_population(self):
        if self.pickup == False:
            return self.w_0
        elif self.pickup == True and self.target == False:
            return self.w_1
            
    def init_run(self):
        #Initiliaze Output layer
        self.N_action = 4
        self.output = np.zeros(self.N_action)
        
        #Initiliaze populations
        self.w_0 = np.random.normal(0,0.1, (64,self.N_action))
        self.w_1 = np.random.normal(0,0.1, (64,self.N_action))
        
        #Initialize Q-values???


       
