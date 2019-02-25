#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:09:23 2019

@author: jagtarsingh
"""

import numpy as np

'''
#################Bias and sensitivity################  
'''  

def ADC2Acc(adc,g,data_num,bias):
    out = np.zeros(6)
    
    
    zero_level_voltage_acc = np.array([1645,1616,1650])
    zero_level_voltage_acc = np.array([1645.16129032, 1612.90322581, 1612.90322581])

    vref = 3300.0#mV
    max_adc_10bit = 1023.0
    sensitivity_acc = 300.0#mV/g
    sensitivity_acc = 338.0
    out[:3] = (((adc[:3]/max_adc_10bit)*vref)-zero_level_voltage_acc)*g/sensitivity_acc
    out[2] = -out[2]

    #gyroscope adjustments
    zero_level_voltage_gyro = 1194.0#mV
    zero_level_voltage_gyro = np.array([1193.65981885,
       1200.60987457, 1195.58090231])
    if data_num !=1 and data_num !=4 and data_num !=2:
        
        zero_level_voltage_gyro = np.array([1193.1483871, 1202.3516129, 1210.4516129])
        zero_level_voltage_acc = bias[3:,0]
    else:
        zero_level_voltage_gyro = np.array([1193.5483871 ,1219.35483871, 1212.90322581])
    vref = 3300.0#mV
    max_adc_10bit = 1023.0

    sensitivity_gyro = [3.5,3.5,3.5]

    out[3:] = (((adc[3:]/max_adc_10bit)*vref)-zero_level_voltage_gyro)*(3.14/180)/sensitivity_gyro
            
    return out





def fix_IMU(vals,t_stamp,g,data_num,bias)  :
    fixed_IMU = np.zeros([6,t_stamp.shape[1]])

    for i in range(t_stamp.shape[1]):
        fixed_IMU[:,i] = ADC2Acc(vals[:,i],g,data_num,bias)
        
    fixed_IMU_new = np.copy(fixed_IMU)
    fixed_IMU_new[3,:]= fixed_IMU[4,:]
    fixed_IMU_new[5,:]= fixed_IMU[3,:]
    fixed_IMU_new[4,:]= fixed_IMU[5,:]
    
    
    return fixed_IMU_new