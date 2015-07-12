# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:12:14 2015

@author: tabea
"""

maze = False
pickup = False
target = False

def in_maze(x,y):
    if 50<=x<=60 and 0<=y<50:
        maze = True
    elif 0<=x<=110 and 50<=y<=60:
        maze = True
    else:
        maze = False

def in_pickup(x,y):
    if x<90 and maze==True:
        pickup = False
    else:
        pickup = True
        
def in_target(x,y):
    if x<=20 and maze==True:
        target = True
    else:
        target = False