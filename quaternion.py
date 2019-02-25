#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:13:20 2019

@author: jagtarsingh
"""
import math
import numpy as np
'''
#################Quaternion Operations################  
'''  
#Function for Quaternion Multiplication
def q_mul(q1, q0):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def vec2quat(vec):
    angle = np.linalg.norm(vec)
    if angle == 0:
        return np.array([1,0,0,0])
    axis = vec/angle
    
    return angleAxis2quat(angle,axis)
    
#def vec2quat(vec):
#    """
#    :param vec: (3x2n)
#    :return quat: (4x2n)
#    """
#    theta = np.linalg.norm(vec, axis=0)
#    axis = np.divide(vec, theta).reshape(3)
#    quat = np.zeros(4)
#    quat[0] = np.cos(theta/2.)
#    quat[1:] = axis * np.sin(theta/2.)
#
#    # if norm == 0 would cause NaN or Inf
#    quat[np.isnan(quat)] = 0
#    quat[np.isinf(quat)] = 0
#
#    return quat
    
def quat2vect(quat):
    if quat[0]<0:
        quat2 = -1*quat
    else:
        quat2 = quat
    theta_half = np.arccos(quat2[0])
    if theta_half == 0:
        return np.array([0,0,0])
    return 2*theta_half*quat2[1:]/np.sqrt(1-quat2[0]**2)
    

    
def angleAxis2quat(angle,axis):
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    q = q* math.sin(angle/2.0) 
    q[0] = math.cos(angle/2.0)
    
    return q/np.linalg.norm(q)
    
def quat_inverse(q):
    return np.array([q[0],-q[1],-q[2],-q[3]])/np.linalg.norm(q)

##### quaternion to roll, pitch, yaw #####
def quat2rpy(quat):
    """
    :param quat: (4x12)
    :return row, pitch, yaw:
    """
    q = quat.copy()

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # roll (x-axis)
    r_sin = 2 * (qw * qx + qy * qz)
    r_cos = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(r_sin, r_cos)

    # pitch (y-axis)
    p_sin = 2 * (qw * qy - qz * qx)
    pitch = np.arcsin(p_sin)
    # if p_sin.any() >= 1:
    #     print('p_sin>=1')

    # yaw (z-axis)
    y_sin = 2 * (qw * qz + qx * qy)
    y_cos = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(y_sin, y_cos)

    return roll, pitch, yaw

def gradient_descent(prev_q,sigma_points):
    q_i = sigma_points
    for i in range(100):
        e = np.zeros_like(q_i)
        e_vect = np.zeros([3,6])#for calculating covariance
        q_prev_inv = quat_inverse(prev_q)
        for j in range(6):
            e[:,j] = q_mul(q_i[:,j],q_prev_inv).T
            e_vect[:,j] = quat2vect(e[:,j])
        e_bar = np.mean(e_vect, axis=1)
        e_bar_quat = vec2quat(e_bar)
        q_bar = q_mul(e_bar_quat,prev_q)        
        error = np.linalg.norm(e_bar)
        if error<0.01:
            return e_vect,e_bar,q_bar.T
        prev_q = q_bar
    return e_vect,e_bar,q_bar.T

def quat2rot(q):
    u=np.zeros((3,3))
    u[0,1]=-q[3]
    u[0,2]=q[2]
    u[1,0]=q[3]
    u[1,2]=-q[1]
    u[2,0]=-q[2]
    u[2,1]=q[1]   
    q_vec = np.reshape(q[1:4],(3,1))
    R=((q[0]**2)-np.matmul(np.transpose(q_vec),q_vec))*np.eye(3)+(2*q[0]*u)+2*np.matmul(q_vec,np.transpose(q_vec))
    return R

def rot2euler(R):   
    roll=np.arctan2(R[2,1],R[2,2])
    pitch=np.arctan2((-1)*np.float64(R[2,0]),(((R[2,1])**2+(R[2,2])**2)**0.5))
    yaw=np.arctan2(R[1,0],R[0,0])
    return roll,pitch,yaw

            
'''
####### Quaternion Operations End Here ###############
'''