#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:11:16 2019

@author: jagtarsingh
"""


import scipy.io as sio
import numpy as np
import random
import math
import pdb
from matplotlib import pyplot as plt
from quaternion import *
from ukf_model import *
from data_loading import *
from scale_imu_data import *

 

def estimate_rot(data_num=2):
    
    g = -9.81#m/sec^2
    vals,t_stamp,rots,t_vicon = load_data(data_num)
    
    bias = np.mean(vals[:,:1],1).reshape(6,1)
    bias = bias*3300.0/1023.0#convert ADC values to Voltage(mV)
    z_measured = fix_IMU(vals,t_stamp,g,data_num,bias)#fix IMU data
    
    P = np.eye(3)
    Q = np.diag([100,100,90])#Process Noise Covariance
    R = np.diag([300,300,200])#Measurement Noise Covariance
#    R = np.diag([100,100,100])
    
    
    state_vec = np.zeros([4,1])
    state_vec[0,0] = 1
    
    euler = np.zeros([3,t_stamp.shape[1]])
    for i in range(t_stamp.shape[1]):

        z_k = z_measured[:,i]
        if i<t_stamp.shape[1]-1:
            
            dt =  t_stamp[0,i+1]-t_stamp[0,i]
        else:
            dt = dt
        Y = make_sigma_points_and_transform(state_vec, Q, P,dt,z_k)
        e_i,Y_mean = compute_mean(Y, state_vec)
        W_i, cov_P = compute_covariance(e_i,Y_mean,Y)
        Z = compute_Z(Y)
        Z_mean = np.mean(Z,axis = 1)
        C = Z-Z_mean.reshape(3,1)
        P_zz = np.matmul(C,C.T)/6.0
        P_xz = np.matmul(W_i,C.T)/6.0
        P_vv = P_zz+R
        K = np.matmul(P_xz,np.linalg.inv(P_vv))
        v_k = z_k[:3]-Z_mean
        K_vk = np.matmul(K,v_k.reshape(3,1))
        quat_new = q_mul(Y_mean.reshape(4,1),vec2quat(K_vk))
        state_vec = quat_new
#        Rot = quat2rot(quat_new)
#        euler[:,i] = rot2euler(Rot)
        euler[:,i] = quat2rpy(quat_new)
        P = cov_P - np.matmul(np.matmul(K,P_vv),K.T)
    
    roll = euler[0,:]
    pitch = euler[1,:]
    yaw = euler[2,:]
    return roll,pitch,yaw













