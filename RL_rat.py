# -*- coding: utf-8 -*-
"""
Models of Higher Brain Function - Reinforcement Learning Project

@author: Oliver Eberle
@author: Tabea Kossen
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class Rat:

    def __init__(self,N_action=4,lmbd=0.95):
        '''Constructor: initialize rat
        
        :param N_action: specifies number of directions the rat can take
                         default: 4
        :type integer
        
        :param lmbd: lambda which specifies the decay rate of the eligibility trace
                     default: 0.95; 0 for a rat without eligibility trace
                     value should be between 0 and 1
        :type float'''
        
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
        
        # initialize some parameters
        self.epsilon = 0.9
        self.gamma = 0.95
        self.lmbd = lmbd

        # number of direction the rat can run
        self.N_action = N_action
        # directions the rat can take
        self.action_arr = np.linspace(2*np.pi/self.N_action,2*np.pi,self.N_action)
        
        # initialize weights
        self.w_0 = np.random.normal(0,0.1, (64,self.N_action))
        self.w_1 = np.random.normal(0,0.1, (64,self.N_action))        
        
        # initialize eligibility trace
        self.e_0 = np.zeros((64,self.N_action))
        self.e_1 = np.zeros((64,self.N_action)) 

        # set parameters for first run
        self._init_run()
     
     
    def _init_run(self):
        '''Initialize parameters so that the rat can start the a new run'''
        
        # initialize output layer, Q-values and rate
        self.output = np.zeros(self.N_action)
        self.Q = np.zeros((self.N_action))
        self.rates = np.zeros(self.cell_centers.shape[0])  
        
        # initialize current action, index of current action and index of old action
        self.action = 0
        self.action_idx = 0
        self.action_old = 0
        
        # set rat to start position and initialize old_position and a trajectory 
        # array which saves all the positions the rat has been until the reward
        # was reaches
        self.rat_position = np.array([55,0])
        self.old_position = np.copy(self.rat_position)
        self.trajectory = np.copy(self.rat_position)
                
        # initialize variables that help keep track if rat is still in maze or has
        # reaches the pickup or target area yet
        self.maze = False
        self.target = False
        self.pickup = False
        
        # initialize reward and beta which specifies which population of neurons is 
        # active at the moment
        self.reward = 0
        self.beta = 0
        
        # epsilon decays with every trial the rat runs
        if self.epsilon>=0.1:
            self.epsilon *= 0.8
            
        self._update_Q(self.w_0)  
 
       
    def run(self,nr_rats=1,nr_runs=1):
        '''Is called by the user in order to let the rat run
        
        :param nr_rats: specifies how many rats should run
                        default: 1
        :type integer
        
        :param nr_runs: specifies how many trials the rat(s) should run
                        default: 1
        :type integer'''
        
        self.nr_rats = nr_rats
        self.nr_runs = nr_runs 
        # the latencies of each run are stored in a matrix
        self.latencies = np.zeros((nr_rats,nr_runs))
        
        # for every rat and every trial
        for i in range(nr_rats):
            for j in range(nr_runs):
                
                 # initialize parameters
                 self._init_run()
                 print('Rat %s, Run %s'%(i+1,j+1))
                 
                 # run and store latencies
                 latency = self._run_trial()
                 self.latencies[i][j] = latency
                 
            if nr_rats!=1:
                self.reset()
                
                
    def _run_trial(self):
        '''run a single trial until the rat reaches the reward position
         
        :return: latency; time that the rat needed to get to the reward'''

        # initialize the latency (time to reach the target) for this trial
        latency = 0.
        
        # SARSA algorithm
        self._choose_action()
        self._update_reward()

        while self.target == False:
            # update trajectory
            self.trajectory = np.c_[self.trajectory, self.rat_position]
            
            # save old position and action
            self.old_position = np.copy(self.rat_position)
            self.action_old = np.copy(self.action_idx)

            # update the position
            self._update_position()
            
            self._choose_action()     
            self._update_reward()
            
            # update weights according to SARSA
            w = self._sarsa()
            
            # update Q
            if isinstance(w,np.ndarray):
                self._update_Q(w)
            
            latency += 1
        
        # put last position in trajectory
        self.trajectory = np.c_[self.trajectory, self.rat_position]
      
        return latency   
    
    
    def _choose_action(self):
        '''choose the next action the rat will do'''
        
        # in 1-epsilon of the cases the action is chosen according to the maximal 
        # Q-value
        if random.random()<(1-self.epsilon):
            self.action_idx = np.argmax(self.Q)
            self.action = self.action_arr[self.action_idx]
            
        # otherwise the action is chosen randomly
        else:
            self.action_idx = random.randint(0,self.N_action-1)
            self.action = self.action_arr[self.action_idx]
        
        
    def _update_reward(self):
        '''update the current reward according to state'''
        
        # if the target is reaches after visiting the pickup area, a reward of 20
        # is given
        if self.target==True and self.pickup==True:
            self.reward = 20
        # if the rat is not in the maze anymore, i.e. it hit a wall, a reward of -1
        # is given
        elif self.maze==False:
            self.reward = -1
        # otherwise no reward
        else:
            self.reward = 0        
        
        
    def _update_position(self):        
        '''update the position according to the chosen current action with a 
        velocity drawn from a normal distribution'''
        
        # direction in which the rat is going next
        run_vector = np.array([np.cos(self.action), np.sin(self.action)])\
                    /(np.linalg.norm([np.cos(self.action), np.sin(self.action)]))
        
        # calculate new position
        self.rat_position += run_vector*self._get_velocity()
        
        # check if rat is still in maze, has reached the pickup or target area
        self._in_maze(self.rat_position[0], self.rat_position[1])
        self._in_target(self.rat_position[0], self.rat_position[1])
        self._in_pickup(self.rat_position[0], self.rat_position[1])
        
        # if the rat is not in the maze anymore, it receives a reward of -1 and is
        # set back to the last position again
        if self.maze == False:
            self._update_reward()
            self.rat_position = np.copy(self.old_position)
            self.old_position = np.copy(self.trajectory[:,-2])
       
       
    def _get_velocity(self):
        '''determine the current velocity of the rat
        
        :return: velocity drawn from a normal distribution'''
        
        return np.random.normal(3, 1.5)       
                
                
    def _in_maze(self,x,y):
        '''check if rat is still in the maze
        
        :param x: position x in the maze
        :type float
        
        :param y: position y in the maze
        :type float'''
        
        if 50<=x<=60 and 0<=y<50:
            self.maze = True
        elif 0<=x<=110 and 50<=y<=60:
            self.maze = True
        else:
            self.maze = False
    
    
    def _in_pickup(self,x,y):
        '''check if rat has been in the pickup area yet
        
        :param x: position x in the maze
        :type float
        
        :param y: position y in the maze
        :type float'''
        
        # check if rat is still in maze
        self._in_maze(x,y)
        
        # if pickup is True, it stays True for the current trial
        if self.pickup==False:
            if x>90 and self.maze==True:
                self.pickup = True
                # when the rat reaches the pickup are, the other population has to 
                # be activated
                self.beta = 1
            else:
                self.pickup = False
            
            
    def _in_target(self,x,y):
        '''check if rat has reached the target area yet
        
        :param x: position x in the maze
        :type float
        
        :param y: position y in the maze
        :type float'''
        
        # check if rat is still in maze
        self._in_maze(x,y)
        
        if x<=20 and self.maze==True and self.pickup==True:
            self.target = True
        else:
            self.target = False         
  

    def _sarsa(self,gamma=0.95,eta=0.1):
        '''implements the SARSA update rule in order to update the weights
        
        :param gamma: reward discount factor, should be between 0 and 1
                      default: 0.95
                      
        :param eta: learning rate, should be small (<<1)
                    default: 0.1
                    
        :return: updated weights'''
        
        # find closest place cell center from current rat position
        min_idx = np.argmin(np.linalg.norm(self.cell_centers- \
                  self.rat_position,axis=1))
        # and from old rat position
        min_idx_old = np.argmin(np.linalg.norm(self.cell_centers- \
                  self.old_position,axis=1))
        
        # choose population
        # if pickup is False, use population 0
        if self.pickup == False:
                        
            # update the eligibility trace
            self.e_0 *= self.gamma * self.lmbd
            self.e_0[min_idx_old,self.action_old] += 1.
            
            # update weights
            self.w_0 += eta*(self.reward+gamma*self.w_0[min_idx,self.action_idx]- \
                        self.w_0[min_idx_old,self.action_old])*self.e_0
                   
            return self.w_0
            
        # if pickup is True and the target is not reached yet, population 1 is 
        # activated
        elif self.pickup == True and self.target == False:
                        
            # update the eligibility trace
            self.e_1 *= self.gamma * self.lmbd
            self.e_1[min_idx_old,self.action_old] += 1.
            
            # update weights
            self.w_1 += eta*(self.reward+gamma*self.w_1[min_idx,self.action_idx]- \
                        self.w_1[min_idx_old,self.action_old])*self.e_1
                   
            return self.w_1

        
    def _update_Q(self,w):
        '''update the Q-values according to the weights and the firing rates'''
        
        self._update_firing_rate()
        self.Q = np.dot(w.T,self.rates)


    def _update_firing_rate(self, sigma = 5):
        '''update the firing rate according to the given formular'''
        
        # if alpha=beta
        if self.beta == self.pickup:
            for j in range(self.cell_centers.shape[0]):
                self.rates[j] = np.exp(-((np.linalg.norm(self.rat_position - \
                self.cell_centers[j,:])) ** 2) / (2 * (sigma ** 2)))
        # 0 otherwise
        else:
            self.rates = np.zeros_like(self.rates)
            
          
    def reset(self):
        '''reset the weights; the rat unlearns the task'''
        
        # new random weights
        self.w_0 = np.random.normal(0,0.1, (64,self.N_action))
        self.w_1 = np.random.normal(0,0.1, (64,self.N_action))
        
        # initialize eligibility trace
        self.e_0 = np.zeros((64,self.N_action))
        self.e_1 = np.zeros((64,self.N_action)) 
        
        # reset epsilon
        self.epsilon = 0.9
        
        
    def plot_maze(self):
        '''plot the maze with the current/last trajectory'''
        
        plt.figure(figsize=(5.5,3))
        plt.plot(self.cell_centers[:,0],self.cell_centers[:,1],'ko',markersize=0.8)
        ax = plt.gca()
        ax.axhspan(50,60,alpha=0.15,color='gray')
        ax.axhspan(50,60,xmin=9/11,alpha=0.05,color='gray')
        ax.axhspan(50,60,xmax=2/11,alpha=0.15,color='gray')
        ax.axvspan(50,60,ymax=5/6,alpha=0.15,color='gray')
        ax.axis('off')
        plt.xlim(0,110)
        plt.ylim(0,60)
        # plot rats' path
        plt.plot(self.trajectory[0,:],self.trajectory[1,:])
        plt.show()


    def plot_latencies(self):
        '''plot the latencies across trials averaged over the rats'''
        
        plt.plot(np.mean(self.latencies,axis=0))
        plt.xlabel('Number of trial')
        plt.ylabel('Averaged latency')
        plt.title('Latencies across %s trials averaged over %s rats'%(self.nr_runs,self.nr_rats))
        plt.show()
       