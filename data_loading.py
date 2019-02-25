#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:36:54 2019

@author: jagtarsingh
"""
import scipy.io as sio

def load_data(data_num = 1):

    IMU = sio.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    vals = IMU['vals']#IMU Data
    t_imu = IMU['ts']
    
    vicon = sio.loadmat('./vicon/viconRot'+str(data_num)+'.mat')
    rots = vicon['rots']#Vicon Data
    t_vicon = vicon['ts']
    
    return vals,t_imu,rots,t_vicon