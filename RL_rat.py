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
        
        self.action_idx = 0
        self.epsilon = 0.1
        
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
        plt.figure(figsize=(5.5,3))
        plt.plot(self.cell_centers[:,0],self.cell_centers[:,1],'ko',markersize=0.8)
        ax = plt.gca()
        ax.axhspan(50,60,alpha=0.15,color='gray')
        ax.axvspan(50,60,ymax=5/6,alpha=0.15,color='gray')
        ax.axis('off')
        plt.xlim(0,110)
        plt.ylim(0,60)
        plt.show()
        
    def update_firing_rate(self, sigma = 5):
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
        
        #Initialize Q-values
        self.Q = np.zeros((N_action))
        
        self.action_arr = np.linspace(2*np.pi/N_action,2*np.pi,N_action)
        self.action = 0

        
        
    def run(self,nr_steps=10,nr_runs=1):
        self.trajectory = np.zeros((nr_steps,2))
        
    def run_trial(self):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.
        """
        # Initialize the latency (time to reach the target) for this trial
        latency = 0.
            
        # Choose a random initial position and make sure that it is not in the wall.
        # Needed here:
        # self.x_position, self.y_position, self._is_wall
        self.x_position = random.randint(0,self.N)
        self.y_position = random.randint(0,self.N)    
        while self._is_wall(self.x_position,self.y_position):
            self.x_position = random.randint(0,self.N)
            self.y_position = random.randint(0,self.N) 
        
        # Run the trial by choosing an action and repeatedly applying SARSA
        # until the reward has been reached.
        # Needed here:
        # self._choose_action, self._arrived,  self._update_state, self._update_Q
        self.choose_action()
        while self.in_target() == False:
            self.update_state()
            self.choose_action()
            w = self.choose_population()
            self.update_Q(w)
            
            latency += 1
      
        return latency
        
    def update_Q(self,w):
        self.update_firing_rate()
        self.Q = np.dot(w.T,self.rates)

    def choose_action(self):
        self.action_old = self.action
        if random.random()<(1-self.epsilon):
            self.action_idx = np.argmax(self.Q)
            self.action = self.action_arr[action_idx]
        else:
            self.action_idx = random.randint(0,N_action-1)
            self.action = self.action_arr[action_idx]

       
