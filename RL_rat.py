# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:12:14 2015

@author: tabea
"""

import numpy as np
import matplotlib.pyplot as plt
import random

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
        
        self.epsilon = 0.9
        
        self.gamma = 0.95
        self.lmbd = 0.95

        self.N_action = 4
        self.w_0 = np.random.normal(0,0.1, (64,self.N_action))
        self.w_1 = np.random.normal(0,0.1, (64,self.N_action))

        self.init_run()

        self.update_Q(self.w_0)        
        
        
                
    def in_maze(self,x,y):
        if 50<=x<=60 and 0<=y<50:
            self.maze = True
        elif 0<=x<=110 and 50<=y<=60:
            self.maze = True
        else:
            self.maze = False
    
    def in_pickup(self,x,y):
        self.in_maze(x,y)
        if self.pickup==False:
            if x>90 and self.maze==True:
                self.pickup = True
            else:
                self.pickup = False
            
    def in_target(self,x,y):
        self.in_maze(x,y)
        if x<=20 and self.maze==True and self.pickup==True:
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
        plt.plot([20,20],[50,60],'b')
        
        plt.plot(self.trajectory[0,:],self.trajectory[1,:])
        plt.show()
        
    def update_firing_rate(self, sigma = 5):
        if self.beta == self.pickup:
            for j in range(self.cell_centers.shape[0]):
                self.rates[j] = np.exp(-((np.linalg.norm(self.rat_position - \
                self.cell_centers[j,:])) ** 2) / (2 * (sigma ** 2)))

        else:
            self.rates = np.zeros_like(self.rates)
            
    def get_velocity(self):
        return np.random.normal(3, 1.5)

            
    def init_run(self):
        #Initialize Output layer
        self.N_action = 4
        self.output = np.zeros(self.N_action)
        
        #Initialize Q-values
        self.Q = np.zeros((self.N_action))
        
        self.action_arr = np.linspace(2*np.pi/self.N_action,2*np.pi,self.N_action)
        self.action = 0
        self.action_idx = 0
        self.action_old = 0
        
        self.latency_list = []

        self.rat_position = np.array([55,0])
        self.old_position = np.copy(self.rat_position)
        self.trajectory = np.copy(self.rat_position)
        
        self.target = False
        self.pickup = False
        
        self.reward = 0
        
        if self.epsilon>=0.1:
            self.epsilon *= 0.999
        
        
    def run(self,nr_runs=1):
        self.latencies = np.zeros(nr_runs)
        
        for i in range(nr_runs):
             self.init_run()
             print(i)
             
             latency = self.run_trial()
             self.latencies[i] = latency
            
    def plot_latencies(self):
        plt.plot(self.latencies)
        plt.xlabel('Number of trial')
        plt.ylabel('Latency')
        plt.title('Latencies across trials')
        plt.show()
        
        
    def run_trial(self):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.
        """
        # Initialize the latency (time to reach the target) for this trial
        latency = 0.
        
        # Run the trial by choosing an action and repeatedly applying SARSA
        # until the reward has been reached.
        # Needed here:
        # self._choose_action, self._arrived,  self._update_state, self._update_Q
        self.choose_action()
        #r_t+1
        self.update_reward()

        while self.target == False:
            #Update Trajectory
            self.trajectory = np.c_[self.trajectory, self.rat_position]
            
            self.old_position = np.copy(self.rat_position)
            self.action_old = np.copy(self.action_idx)

            
            #s_t+1
            self.rat_position = self.update_position()
            #a_t+1
            self.action_idx, self.action = self.choose_action()     
            #r_t+1
            self.update_reward()
            
            
            #update weights according to SARSA
            w = self.sarsa()
            
            if isinstance(w,np.ndarray):
                self.update_Q(w)
                        
            latency += 1
      
        return latency
        
    def sarsa(self,gamma=0.95,eta=0.1):
        # find closest place cell center 
        min_idx = np.argmin(np.linalg.norm(self.cell_centers- \
                  self.rat_position,axis=1))
        min_idx_old = np.argmin(np.linalg.norm(self.cell_centers- \
                  self.old_position,axis=1))
                  
        # choose population
        if self.pickup == False:
            
            self.w_0[min_idx,self.action_idx] += \
                   eta*(self.reward+gamma* \
                   self.w_0[min_idx,self.action_idx]- \
                   self.w_0[min_idx_old,self.action_old])
                   
            return self.w_0
            
        elif self.pickup == True and self.target == False:
            
            self.w_1[min_idx,self.action_idx] += \
                   eta*(self.reward+gamma* \
                   self.w_1[min_idx,self.action_idx]- \
                   self.w_1[min_idx_old,self.action_old]) 
                   
            return self.w_1


    def update_reward(self):
        if self.target==True and self.pickup==True:
            self.reward += 20
        elif self.maze==False:
            self.reward -= 1
        
    def update_Q(self,w):
        self.update_firing_rate()
        self.Q = np.dot(w.T,self.rates)

    def choose_action(self):
        #returns self.action_idx and self.action
        if random.random()<(1-self.epsilon):
           # self.action_idx = np.argmax(self.Q)
            #self.action = self.action_arr[self.action_idx]
            return np.argmax(self.Q), self.action_arr[self.action_idx]
        else:
            #self.action_idx = random.randint(0,self.N_action-1)
            #self.action = self.action_arr[self.action_idx]
            return random.randint(0,self.N_action-1), self.action_arr[self.action_idx]


        

            
    def update_position(self):        
        run_vector = np.array([np.cos(self.action), np.sin(self.action)])\
                    /(np.linalg.norm([np.cos(self.action), np.sin(self.action)]))
        
        self.rat_position += run_vector*self.get_velocity()
        
        self.in_maze(self.rat_position[0], self.rat_position[1])
        self.in_target(self.rat_position[0], self.rat_position[1])
        self.in_pickup(self.rat_position[0], self.rat_position[1])
        

        if self.maze == False:
            self.update_reward()
            self.rat_position = np.copy(self.old_position)
            self.old_position = np.copy(self.trajectory[:,-2])
            return self.rat_position 
            
        else:
            return self.rat_position 
            
    def reset(self):
        """
        Reset weights (and the latency_list).
        
        Instant amnesia -  the agent forgets everything he has learned before    
        """
        self.w_0 = np.random.normal(0,0.1, (64,self.N_action))
        self.w_1 = np.random.normal(0,0.1, (64,self.N_action))
        self.target = False
        self.pickup = False
        self.latency_list = []
        
    def get_weights(self):
        return self.w_0, self.w_1

       
