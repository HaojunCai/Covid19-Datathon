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



# read the data
confimed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
confirmed = pd.read_csv(confimed_url)
death_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
deaths = pd.read_csv(death_url)
recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
recovered = pd.read_csv(recovered_url)


colnames = confirmed.columns.tolist()
start = datetime.datetime.strptime(colnames[4], "%m/%d/%y")
end = datetime.datetime.strptime(colnames[-1], "%m/%d/%y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]

countries = ['Argentina', 'Egypt', 'Nigeria']
population = [44.49*10**6, 98.42*10**6, 195.9*10**6]

# time delta for prediction in days
prediction_delta = 2

def pred_sir(S, I, R, D, days, days_delta):
    
    index = next((i for i, x in enumerate(I) if x), None) 
    print(index)
    
    # set up initial conditions 
    S0 = S[index]
    I0 = I[index]
    R0 = R[index] 
    D0 = D[index]
    init = [S0, I0, R0, D0]
    
    S = S[index:]
    I = I[index:]
    R = R[index:]
    D = D[index:]
    obs = np.transpose([S, I, R, D])
    days = days - index 
    
    sir0 = init
    t = np.linspace(0,days,days+1) 
    
    # sir model 
    def sir_model(sir, t, beta, gamma, mui): 
            N = sum(init)
            dsdt = - beta*sir[0]*sir[1]/N
            didt = beta*sir[0]*sir[1]/N - gamma*sir[1] - mui*sir[1]
            drdt = gamma*sir[1]
            dddt = mui*sir[1]
            dsirdt = [dsdt, didt, drdt, dddt]
            return dsirdt
        
    # integrate 
    def sir_int(t, beta, gamma, mui):
        y = odeint(sir_model, sir0, t, args=(beta, gamma, mui))      
        return y.ravel()
    
    def sir_int_(t, beta, gamma, mui):
        y = odeint(sir_model, sir0, t, args=(beta, gamma, mui))      
        return y
    
    # curve fit 
    vals, cov = opt.curve_fit(sir_int, t, obs.ravel())
    
    # simple error estimator without intrinsic errors in data points
    sigma_ab = np.sqrt(np.diagonal(cov))
    days_pred = days+days_delta
    t_pred = np.linspace(0,days_pred,days_pred+1) 
    y_pred = sir_int_(t_pred, vals[0], vals[1], vals[2])
    y_pred_upper = sir_int_(t_pred, vals[0]-1.96*sigma_ab[0], vals[1]-1.96*sigma_ab[1], vals[2]-1.96*sigma_ab[2])
    y_pred_lower = sir_int_(t_pred, vals[0]+1.96*sigma_ab[0], vals[1]+1.96*sigma_ab[1], vals[2]+1.96*sigma_ab[2])
    

    # plot 
    #plt.plot(t_pred,y_pred[:,1],'r-',label=r'i')
    #plt.plot(t,obs[:,1], 'r--', label=r'i_obs')
    plt.plot(t_pred,y_pred[:,0],'b-',label=r's')
    plt.plot(t,obs[:,0], 'b--', label=r's_obs')
    #plt.plot(t_pred,y_pred[:,2],'g-',label=r'r')
    #plt.plot(t,obs[:,2], 'g--', label=r'r_obs')
    plt.ylabel('response')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()  
    
    return y_pred, y_pred_lower, y_pred_upper

today = date.today()
next_pred_date = today+datetime.timedelta(days=prediction_delta)

file_str = "2day_prediction_" + str(today) + ".csv"
print(file_str)

f = open(file_str,"w+")
        #Province/State,Country/Region,Target/Date,N,low95N,high95N,R,low95R,high95R,D,low95D,high95D,T,low95T,high95T,M,low95M,high95M
f.write("Province/State,Country,Target/Date,N,low95N,high95N,R,low95R,high95R,D,low95D,high95D,T,low95T,high95T,M,low95M,high95M,C,low95C,high95C\n")
    

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
    pred, pred_upper, pred_lower = pred_sir(susceptible,infect,recover, deaths_region, days, prediction_delta)
    print(pred)
    
    # recover 
    recovered_pred = pred[:,2]
    recovered_pred_lower = pred_lower[:,2]
    recovered_pred_lower[recovered_pred_lower<0] = 0
    recovered_pred_upper = pred_upper[:,2]
    
    # death 
    deaths_pred = pred[:,3]
    deaths_pred_lower = pred_lower[:,3]
    deaths_pred_lower[deaths_pred_lower<0] = 0
    deaths_pred_upper = pred_upper[:,3]
    
    # confirmed = infected + recovered + death 
    confirmed_pred = pred[:,1] +  recovered_pred + deaths_pred
    confirmed_pred_lower = pred_lower[:,1] + recovered_pred_lower + deaths_pred_lower
    confirmed_pred_upper = pred_upper[:,1] + recovered_pred_upper + deaths_pred_upper
    
    conf_flag = 0 
    d_flag = 0 
    rec_flag= 0
    
    try: 
        pred_sir(susceptible,infect,recover, deaths_region, days, prediction_delta)
    except: 
        conf_flag = 1 
        d_flag = 1 
        rec_flag= 1
    
    if np.logical_and(d_flag==0, conf_flag == 0):
        next_mortality = deaths_pred[-1]/confirmed_pred[-1]
        next_mortality_lower = deaths_pred_lower[-1]/confirmed_pred_upper[-1]
        next_mortality_upper = deaths_pred_upper[-1]/confirmed_pred_lower[-1]
    else: 
        next_mortality = ''
        next_mortality_lower = ''
        next_mortality_upper = ''
        
        
        
    
    next_pred_date_str = str(next_pred_date)+","
    loc1_str = ","
    loc2_str = str(countries[i]).replace(',', ' ') + ","
    if conf_flag == 0: 
        n_str = str(round(confirmed_pred[-1]))+","+str(round(confirmed_pred_lower[-1]))+","+str(round(confirmed_pred_upper[-1]))+","
    else: 
        n_str = ","+","+","
    
    if rec_flag == 0: 
        r_str = str(round(deaths_pred[-1]))+","+str(round(deaths_pred_lower[-1]))+","+str(round(deaths_pred_upper[-1]))+","
    else: 
        r_str = ","+","+","
    
    if d_flag == 0 :
        d_str = str(round(recovered_pred[-1]))+","+str(round(recovered_pred_lower[-1]))+","+str(round(recovered_pred_upper[-1]))+","  
    else: 
        d_str = ","+","+","
    
    t_str = ","+","+","
    m_str = str(next_mortality)+","+str(next_mortality_lower)+","+str(next_mortality_upper) + ","
    c_str = ","+","+"\n"  
     
    f.write(loc1_str+loc2_str+next_pred_date_str+n_str+r_str+d_str+t_str+m_str+c_str)
        
print("baseline predictions writtern to:"+file_str)    
f.close()

