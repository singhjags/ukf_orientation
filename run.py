#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:52:48 2019

@author: jagtarsingh
"""

from estimate_rot import *
import scipy.io as sio

def get_g_truth_rpy(rots,t_vicon):
    ground_truth = np.zeros([3,t_vicon.shape[1]])
    for i in range(t_vicon.shape[1]):
        ground_truth[:,i] = rot2euler(rots[:,:,i])
        
    return ground_truth


datanum=3

vicon = sio.loadmat('./vicon/viconRot'+str(datanum)+'.mat')
rots = vicon['rots']#Vicon Data
t_vicon = vicon['ts']
g_rpy = get_g_truth_rpy(rots,t_vicon)
roll,pitch ,yaw = estimate_rot(datanum) 
plt.plot(g_rpy[0,:], label = "vicon")
plt.plot(roll[30:], label = "UKF")
plt.legend()
plt.title("Roll")
plt.show()

plt.plot(g_rpy[1,:], label = "vicon")
plt.plot(pitch[30:], label = "UKF")
plt.legend()
plt.title("Pitch")
plt.show()

plt.plot(g_rpy[2,:], label = "vicon")
plt.plot(yaw[30:], label = "UKF")
plt.legend()
plt.title("Yaw")
plt.show()



    


    
           
            
            
 
        
