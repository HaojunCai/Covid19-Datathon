#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:05:34 2020

@author: wangli
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd 
import datetime
from datetime import date


def prediction(S, I, R, days, days_delta):
    
    index = next((i for i, x in enumerate(I) if x), None)
    
    # set up initial conditions 
    S0 = S[index]
    I0 = I[index]
    R0 = R[index] 
    init = [S0,I0,R0]
    
    S = S[index:]
    I = I[index:]
    R = R[index:]
    obs = np.transpose([S, I, R])
    print(days)
    days = days - index  
    print(days)
    def pred(days, beta, gamma): 
        # beta:  transmit rate 
        # gamma: recover rate 
    
        sir0 = init # initial condition  
        
        N = sum(init) # total population 
        
        # time span 
        t = np.linspace(0,days,days+1) 
        
        # SIR model 
        def sir_model(sir,t):
            dsdt = - beta*sir[0]*sir[1]/N
            didt = beta*sir[0]*sir[1]/N - gamma*sir[1]
            drdt = gamma*sir[1]
            dsirdt = [dsdt, didt, drdt]
            return dsirdt
    
        # solve 
        z = odeint(sir_model,sir0,t)
        return z
  
    # loss function
    def loss(x): 
        pred_ = pred(days, x[0], x[1])
        mal = np.sum(np.abs(np.subtract(pred_, obs))) # mean absolute error 
        return mal
   
    # optimization 
    x0 = [0.9, 0.1]
    res = opt.minimize(loss, x0, method = 'Nelder-Mead')
    
    days_2 = days + days_delta
    z = pred(days_2, res.x[0], res.x[1])
    print(len(z[:,1]))
    t = np.linspace(0, days_2, days_2+1)
    print(len(t))
    
    # plot 
    plt.plot(t,z[:,1],'r-',label=r'i')
    plt.plot(t[0:len(t)-days_delta],I, 'r--', label=r'i_obs')
    #plt.plot(t,z[:,0],'b-',label=r's')
    #plt.plot(t[0:len(t)-days_delta],S, 'b--', label=r's_obs')
    #plt.plot(t,z[:,2],'g-',label=r'r')
    #plt.plot(t[0:len(t)-days_delta],R, 'g--', label=r'r_obs')
    plt.ylabel('response')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()  
    return z 
   