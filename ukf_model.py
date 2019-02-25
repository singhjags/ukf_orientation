#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:25:32 2019

@author: jagtarsingh
"""
import numpy as np
from quaternion import *
#Function for Creatig Sigma Points and Transforming them to Y
def make_sigma_points_and_transform(state_vec,Q, P,dt,IMU_readings):

    S = np.linalg.cholesky(P+Q)
    W_temp = np.sqrt(6)*S
    W = np.concatenate((W_temp, -W_temp), axis=1)#6x12 matrix
    X  = np.zeros([4,2*3])
    angle = np.linalg.norm(IMU_readings[3:])*dt
    if angle!=0:
        axis = IMU_readings[3:]*dt/angle
        q_delta = angleAxis2quat(angle,axis)
        q_delta = q_delta/np.linalg.norm(q_delta)
    else:
        q_delta = np.array([1,0,0,0])

    for k in range(6):        
        X[:,k] = q_mul(state_vec[:,0],vec2quat(W[:,k]))#q_{k-1}*q_W
        X[:,k] = q_mul(X[:,k],q_delta).reshape(4)
        
    return X
    
def compute_mean(transformed_sig_points, prev_q):

    e_i,_,Y_mean = gradient_descent(prev_q,transformed_sig_points)

    return e_i,Y_mean

def compute_covariance(e_i,Y_mean,transformed_sig_points):
    
    W_i = e_i
    cov_P = np.matmul(W_i,W_i.T)/6.0
    
    return W_i, cov_P

def compute_Z(transformed_sig_points):
    Z = np.zeros([4,6])
    

    g_quat = np.array([0,0,0,1])
    for i in range(6):
        Z[:,i] = q_mul(transformed_sig_points[:4,i],q_mul(g_quat,quat_inverse(transformed_sig_points[:4,i]))) ##qgq-1

    
    return Z[1:,:]    