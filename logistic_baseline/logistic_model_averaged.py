import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import scipy.optimize as opt
import numpy as np
import datetime
from datetime import date

average_length = 3
fit_length = 7
prediction_delta = 2

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
    for i in range(incre.shape[0]-1):
        incre[i+1,:] = raw[i+1,:]-raw[i,:]

    new_incre= incre.copy()
    #induce some noise?
    np.random.seed(7)
    var = np.zeros(new_incre.shape)
    for i in range(new_incre.shape[0]):
        if i < int((average_length+1)/2)-1:
            continue
        var[i,:] = np.var(incre[i:i+average_length,:],axis=0)
        new_incre[i,:] += np.random.normal(0,np.sqrt(var[i,:]),var.shape[1])

    #calculate the cumulative value
    new_cumulative = np.zeros((days,incre.shape[1]))
    for i in range(days):
        if i == 0:
            new_cumulative[i,:] = new_incre[i,:]
        else:
            new_cumulative[i,:] = new_cumulative[i-1,:] + new_incre[i,:]
    return new_cumulative


def logistic(x, k1, k2, k3, x0):
    return k1 / (1.+ np.exp(-k2 * (x - x0)))


def pred_logistic(x_train, y_train, x_pred):
    # This prediction uses a logistic curve f(x)=L*k1/(1+exp(-k*(x-x0))) as fit.
    # L is the total population
    
    L = max(y_train)
    x_pred = np.copy(x_pred)/max(x_train)
    x_train = np.copy(x_train)/max(x_train)
    y_train = np.copy(y_train)/L


    popt, pcov = opt.curve_fit(logistic, x_train, y_train, maxfev=100000)

    sigma_ab = np.sqrt(np.diagonal(pcov))

    y_pred = logistic(x_pred, popt[0], popt[1], popt[2], popt[3])
    y_pred_upper = logistic(x_pred, popt[0]-1.96*sigma_ab[0], popt[1]-1.96*sigma_ab[1], popt[2]-1.96*sigma_ab[2], popt[3]-1.96*sigma_ab[3])
    y_pred_lower = logistic(x_pred, popt[0]+1.96*sigma_ab[0], popt[1]+1.96*sigma_ab[1], popt[2]+1.96*sigma_ab[2], popt[3]+1.96*sigma_ab[3])

    if y_pred_lower[-1] <= y_pred[-1]:
        y_pred_lower[-1] = y_pred[-1]
 
    y_pred_lower *= L
    y_pred_upper *= L
    y_pred *= L

    return y_pred, y_pred_lower, y_pred_upper

confimed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
confirmed = pd.read_csv(confimed_url)
death_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
deaths = pd.read_csv(death_url)
recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
recovered = pd.read_csv(recovered_url)


colnames = confirmed.columns.tolist()

start = datetime.datetime.strptime(colnames[4], "%m/%d/%y")
end = datetime.datetime.strptime(colnames[-1], "%m/%d/%y")
dates = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days+1)]


countryRegion_file = pd.read_csv("countries.csv")
countries = list(countryRegion_file.x)
#countries = ['Germany', 'Italy']
#countries = ['Switzerland', 'Italy', 'Germany', 'US',  'Spain', 'Singapore', 'India', 'Iran']
#population = [8.57e6, 60.5e6, 82.8e6, 327.2e6,  46.66e6, 66.44e6, 126.4e6, 7.50e6, 5.85e6, 1380e6, 83.99e6]
#population = [8.57e6, 60.5e6, 82.8e6, 327.2e6,  46.66e6,  5.85e6, 1380e6, 83.99e6]

herd_imm_ratio = 0.66

today = date.today() - datetime.timedelta(days=1)
next_pred_date = today+datetime.timedelta(days=prediction_delta)

file_str = "2day_prediction_" + str(today) + ".csv"
print(file_str)

f = open(file_str,"w+")
f.write("Province/State,Country,Target/Date,N,low95N,high95N,R,low95R,high95R,D,low95D,high95D,T,low95T,high95T,M,low95M,high95M,C,low95C,high95C\n")


for i in range(len(countries)):
    print(i)
    confirmed_region = confirmed.loc[confirmed['Country/Region'] == countries[i]]
    deaths_region = deaths.loc[deaths['Country/Region'] == countries[i]]
    recovered_region = recovered.loc[recovered['Country/Region'] == countries[i]]
    
 
    confirmed_region = np.asarray([float(sum(confirmed_region[colnames[4+i]])) for i in range(len(dates))])
    deaths_region = np.asarray([float(sum(deaths_region[colnames[4+i]])) for i in range(len(dates))])
    recovered_region = np.asarray([float(sum(recovered_region[colnames[4+i]])) for i in range(len(dates))])

    # smooth 
    confirmed_region = preprocess(confirmed_region).smooth
    deaths_region = preprocess(deaths_region).smooth
    recovered_region = preprocess(recovered_region).smooth
    
    # choose the data 
    date_generated = dates[-fit_length:]
    confirmed_region = confirmed_region[-fit_length:]
    deaths_region = deaths_region[-fit_length:]
    recovered_region = recovered_region[-fit_length:]
    #print(confirmed_region)

    days = np.linspace(0, len(date_generated)-1, len(date_generated))
    days_prediction = np.linspace(0, len(date_generated)-1+prediction_delta, len(date_generated)+prediction_delta)

    start = date_generated[0]
    end =  date_generated[-1]
    dates_prediction = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days+prediction_delta+1)]

    confirmed_pred, confirmed_pred_lower, confirmed_pred_upper = pred_logistic(days, confirmed_region, days_prediction)#, population[i])
    deaths_pred, deaths_pred_lower, deaths_pred_upper = pred_logistic(days, deaths_region, days_prediction)#, population[i])
    recovered_pred, recovered_pred_lower, recovered_pred_upper = pred_logistic(days, recovered_region, days_prediction)#, population[i])

    fig, ax = plt.subplots()
    plt.title('%s'%countries[i])
    plt.plot(date_generated, confirmed_region, 'o', label = 'confirmed cases')
    plt.plot(date_generated, deaths_region, '^', label = 'deaths')
    plt.plot(date_generated, recovered_region, 's', label = 'recovered')
    plt.plot(dates_prediction, confirmed_pred, 'grey', label = 'prediction')
    plt.plot(dates_prediction, deaths_pred, 'grey')
    plt.plot(dates_prediction, recovered_pred, 'grey')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xlim([min(dates_prediction), max(dates_prediction)])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
    plt.xticks(rotation = 45)
    plt.legend(loc = 2)
    plt.ylabel('cases')
    plt.tight_layout()
    plt.show()
    
    next_pred_date_str = str(next_pred_date)+","
    loc1_str = ","
    loc2_str = str(countries[i]).replace(',', ' ') + ","
        
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
    
    
    if conf_flag == 0:
        #n_str = str(round(confirmed_pred[-1]))+","+str(round(confirmed_pred_lower[-1]))+","+str(round(confirmed_pred_upper[-1]))+","
        n_str = str(round(confirmed_pred[-1]))+","+","+","
    else:
        n_str = ","+","+","

    if rec_flag == 0:
        r_str = str(round(recovered_pred[-1]))+","+","+","
    else:
        r_str = ","+","+","

    if d_flag == 0:
        d_str = str(round(deaths_pred[-1]))+","+","+","
    else:
        d_str = ","+","+","

    t_str = ","+","+","
    m_str = str(next_mortality)+","+","+","
    c_str =  ","+","+","+ "\n"

    f.write(loc1_str+loc2_str+next_pred_date_str+n_str+r_str+d_str+t_str+m_str+c_str)

print("baseline predictions writtern to:"+file_str)
f.close()
