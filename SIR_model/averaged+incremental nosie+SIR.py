"""
Created on Wed Apr 15 11:35:56 2020

@author: wangli
"""
import sys
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

#countries = ['Switzerland', 'Italy', 'Germany', 'US',  'Spain', 'United Kingdom', 'Japan', 'China_Hong Kong', 'Singapore', 'India', 'Iran']
countries = ['Switzerland', 'Italy', 'Germany', 'US',  'Spain', 'Singapore', 'India', 'Iran','Japan']
#population = [8.57e6, 60.5e6, 82.8e6, 327.2e6,  46.66e6, 66.44e6, 126.4e6, 7.50e6, 5.85e6, 1380e6, 83.99e6]
population = [8.57e6, 60.5e6, 82.8e6, 327.2e6,  46.66e6,  5.85e6, 1380e6, 83.99e6, 126.4e6]
#countries = ['Italy']
#population = [82.8e6]

# time delta for prediction in days
prediction_delta = int(sys.argv[1])
average_length = 5
fit_length = 7

class preprocess():
    def __init__(self,DataList):
        self.raw = DataList
        self.average_length = average_length
        self.average()
    def average(self):
        self.smooth = (self.raw).copy()
        for i in range(len(self.raw)-int((average_length+1)/2)):
            self.smooth[i+int((average_length+1)/2)-1] = np.mean(self.raw[i:i+average_length-1])




def increment(raw, days):
    incre = raw.copy()
    for i in range(len(incre)-1):
        incre[i+1] = raw[i+1]-raw[i]

    new_incre= incre.copy()
    #induce some noise?
    np.random.seed(7)
    var = np.zeros(len(new_incre))
    for i in range(len(new_incre)):
        if i < int((average_length+1)/2)-1:
            continue
        var[i] = np.var(incre[i:i+average_length])
        new_incre[i] += np.random.normal(0,np.sqrt(var[i]),1)



    #calculate the cumulative value
    new_cumulative = np.zeros((days))

    for i in range(days):
        if i == 0:
            new_cumulative[i] = new_incre[i]
        else:
            new_cumulative[i] = new_cumulative[i-1] + new_incre[i]

    return new_cumulative



def pred_sir(S, I, R, D, days, days_delta):

    #index = next((i for i, x in enumerate(I) if x), None)
    #print(index)
    index = 0
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


    sir0 = init

    t = np.arange(0,days)

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
    extended_days_pred = days_pred+int((average_length+1)/2)
    t_pred_extended = np.linspace(0,extended_days_pred,extended_days_pred+1)

    y_pred = sir_int_(t_pred_extended, vals[0], vals[1], vals[2])
    for i in range(4):
        if i == 0 :
            new_pred = increment(y_pred[:,0],days_pred)
        else:
            new_pred = np.vstack((new_pred,increment(y_pred[:,i],days_pred)))

    y_pred_upper = sir_int_(t_pred, vals[0]-1.96*sigma_ab[0], vals[1]-1.96*sigma_ab[1], vals[2]-1.96*sigma_ab[2])
    for i in range(4):
        if i == 0 :
            new_upper = increment(y_pred_upper[:,i],days_pred)
        else:
            new_upper = np.vstack((new_upper,increment(y_pred_upper[:,i],days_pred)))
    print(new_upper.shape)
    y_pred_lower = sir_int_(t_pred, vals[0]+1.96*sigma_ab[0], vals[1]+1.96*sigma_ab[1], vals[2]+1.96*sigma_ab[2])
    for i in range(4):
        if i == 0 :
            new_lower = increment(y_pred_lower[:,i],days_pred)
        else:
            new_lower = np.vstack((new_lower,increment(y_pred_lower[:,i],days_pred)))



    # plot
    #plt.plot(t_pred,y_pred[:,1],'r-',label=r'i')
    #plt.plot(t,obs[:,1], 'r--', label=r'i_obs')
    plt.plot(y_pred[:,1],label=r'sa')
    plt.plot(new_pred[:,1],'b-',label=r's')
    plt.plot(t,obs[:,1], 'b--', label=r's_obs')
    #plt.plot(t_pred,y_pred[:,2],'g-',label=r'r')
    #plt.plot(t,obs[:,2], 'g--', label=r'r_obs')
    plt.ylabel('response')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()

    return new_pred, new_lower, new_upper



########################################################
today = date.today()
next_pred_date = today+datetime.timedelta(days=prediction_delta)


file_str = "./prediction/"+str(prediction_delta)+"day_prediction_" + str(today) + ".csv"
print(file_str)

f = open(file_str,"w+")
        #Province/State,Country/Region,Target/Date,N,low95N,high95N,R,low95R,high95R,D,low95D,high95D,T,low95T,high95T,M,low95M,high95M
f.write("Province/State,Country,Target/Date,N,low95N,high95N,R,low95R,high95R,D,low95D,high95D,T,low95T,high95T,M,low95M,high95M,C,low95C,high95C\n")



for i in range(len(countries)):
    confirmed_region = confirmed.loc[confirmed['Country/Region'] == countries[i]]
    deaths_region = deaths.loc[deaths['Country/Region'] == countries[i]]
    recovered_region = recovered.loc[recovered['Country/Region'] == countries[i]]


    start = datetime.datetime.strptime(colnames[4], "%m/%d/%y")
    end = datetime.datetime.strptime(colnames[-1], "%m/%d/%y")
    date_generated = [datetime.timedelta(days=x) for x in range(0, (end-start).days)]


    confirmed_region = np.asarray([float(confirmed_region[colnames[4+i+1]]) for i in range(len(date_generated))])
    smooth_confirmed = preprocess(confirmed_region).smooth


    deaths_region = np.asarray([float(deaths_region[colnames[4+i+1]]) for i in range(len(date_generated))])
    smooth_deaths = preprocess(deaths_region).smooth

    recovered_region = np.asarray([float(recovered_region[colnames[4+i+1]]) for i in range(len(date_generated))])
    smooth_recovered = preprocess(recovered_region).smooth
    print
    susceptible = population[i] - smooth_confirmed
    infect = smooth_confirmed - smooth_recovered - smooth_deaths
    recover = smooth_recovered

    days = fit_length
    pred, pred_upper, pred_lower = pred_sir(susceptible[-days:],infect[-days:],recover[-days:], deaths_region[-days:], days, prediction_delta)


    # recover
    recovered_pred = pred[2,:]
    recovered_pred_lower = pred_lower[2,:]
    recovered_pred_lower[recovered_pred_lower<0] = 0
    recovered_pred_upper = pred_upper[2,:]

    # death
    deaths_pred = pred[3,:]
    deaths_pred_lower = pred_lower[3,:]
    deaths_pred_lower[deaths_pred_lower<0] = 0
    deaths_pred_upper = pred_upper[3,:]

    # confirmed = infected + recovered + death
    confirmed_pred = pred[1,:] +  recovered_pred + deaths_pred
    confirmed_pred_lower = pred_lower[1,:] + recovered_pred_lower + deaths_pred_lower
    confirmed_pred_upper = pred_upper[1,:] + recovered_pred_upper + deaths_pred_upper

    conf_flag = 0
    d_flag = 0
    rec_flag= 0


    if np.logical_and(d_flag==0, conf_flag == 0):
        next_mortality = deaths_pred[-1]/confirmed_pred[-1]
        next_mortality_lower = deaths_pred_lower[-1]/confirmed_pred_upper[-1]
        next_mortality_upper = deaths_pred_upper[-1]/confirmed_pred_lower[-1]
    else:
        next_mortality = ''
        next_mortality_lower = ''
        next_mortality_upper = ''


    print(confirmed_pred[-1])
    print(confirmed_pred_lower[-1])
    print(confirmed_pred_upper[-1])

    next_pred_date_str = str(next_pred_date)+","
    loc1_str = ","
    loc2_str = str(countries[i]).replace(',', ' ') + ","
    if conf_flag == 0:
        n_str = str(round(confirmed_pred[-1]))+","+str(round(confirmed_pred_lower[-1]))+","+str(round(confirmed_pred_upper[-1]))+","
    else:
        n_str = ","+","+","

    if d_flag == 0:
        d_str = str(round(deaths_pred[-1]))+","+str(round(deaths_pred_lower[-1]))+","+str(round(deaths_pred_upper[-1]))+","
    else:
        d_str = ","+","+","

    if rec_flag == 0 :
        r_str = str(round(recovered_pred[-1]))+","+str(round(recovered_pred_lower[-1]))+","+str(round(recovered_pred_upper[-1]))+","
    else:
        r=r_str = ","+","+","

    t_str = ","+","+","
    m_str = str(next_mortality)+","+str(next_mortality_lower)+","+str(next_mortality_upper) + ","
    c_str = ","+","+"\n"
    print(loc1_str+loc2_str+next_pred_date_str+n_str+r_str+d_str+t_str+m_str+c_str)
    f.write(loc1_str+loc2_str+next_pred_date_str+n_str+r_str+d_str+t_str+m_str+c_str)


print("baseline predictions writtern to:"+file_str)
f.close()
