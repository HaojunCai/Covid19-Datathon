"""
Created on Wed Apr 15 11:35:56 2020

@author: wangli
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd 
import datetime
from datetime import date
import function as func


# read the data
confirmed = pd.read_csv('data/time_series_covid19_confirmed_global.csv')
deaths = pd.read_csv('data/time_series_covid19_deaths_global.csv')
recovered = pd.read_csv('data/time_series_covid19_recovered_global.csv')

colnames = confirmed.columns.tolist()

start = datetime.datetime.strptime(colnames[4], "%m/%d/%y")
end = datetime.datetime.strptime(colnames[-1], "%m/%d/%y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]

countries = ['Italy', 'Germany']
population = [60.5*10**6, 82.8*10**6]

# time delta for prediction in days
prediction_delta = 2


for i in range(len(countries)):
    confirmed_region = confirmed.loc[confirmed['Country/Region'] == countries[i]]
    deaths_region = deaths.loc[deaths['Country/Region'] == countries[i]]
    recovered_region = recovered.loc[recovered['Country/Region'] == countries[i]]
    
    confirmed_region = np.asarray([float(confirmed_region[colnames[4+i+1]]) for i in range(len(date_generated))])
    deaths_region = np.asarray([float(deaths_region[colnames[4+i+1]]) for i in range(len(date_generated))])
    recovered_region = np.asarray([float(recovered_region[colnames[4+i+1]]) for i in range(len(date_generated))])
    
    susceptible = population[i] - confirmed_region
    infect = confirmed_region - recovered_region - deaths_region
    recover = recovered_region
    
    days = len(recovered_region)-1
    func.prediction(susceptible,infect,recover, days, prediction_delta)

