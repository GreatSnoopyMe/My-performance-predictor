from cmath import exp, log
import numpy as np
from scipy.optimize import leastsq
from curve_functions import pow3, pow4, log_power, weibull, mmf, janoschek, ilog2, exp3, exp4, dr_hill_zero_background


all_curve_func = [pow3, pow4, log_power, weibull, mmf, janoschek, ilog2, exp3, exp4, dr_hill_zero_background]


# first get all the x,y
y_train = np.loadtxt('learning_curve.txt')
x_train = np.arange(50)+1

# first get all the theta
lowerbound = np.max(y_train)
popts = {}
convergency = {}
# implementation

## pow3
popt, pcov = leastsq(pow3,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
popts["pow3"] = popt
convergency["pow3"] = popt[0]

## pow4
popt, pcov = leastsq(pow4,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf,-np.inf),(1,np.inf,np.inf,np.inf)))
popts["pow4"] = popt
convergency["pow4"] = popt[0]

## log_power
popt, pcov = leastsq(log_power,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
popts["log_power"] = popt
convergency["log_power"] = popt[0]

## weibull
popt, pcov = leastsq(weibull,xdata=x_train,ydata=y_train,bounds=((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)))
popts["weibull"] = popt
if popt[2]<0:
    raise ValueError("Error in weibull, kappa not legal")
if(popt[3])>=1:
    convergency["weibull"] = popt[0]
elif(popt[3]<1 and popt[3]>
0):
    convergency["weibull"] = popt[1]
else:
    raise ValueError("Error in weibull, delta not legal")

## mmf
popt, pcov = leastsq(mmf,xdata=x_train,ydata=y_train,bounds=((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)))
popts["mmf"] = popt
if popt[2]<0:
    raise ValueError("Error in mmf, kappa not legal")
if(popt[3])>=1:
    convergency["mmf"] = popt[0]
elif(popt[3]<1 and popt[3]>0):
    convergency["mmf"] = popt[1]
else:
    raise ValueError("Error in mmf, delta not legal")

## janoschek
popt, pcov = leastsq(janoschek,xdata=x_train,ydata=y_train,bounds=((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)))
popts["janoschek"] = popt
if popt[2]<0:
    raise ValueError("Error in janoschek, kappa not legal")
if(popt[3])>=1:
    convergency["janoschek"] = popt[0]
elif(popt[3]<1 and popt[3]>
0):
    convergency["janoschek"] = popt[1]
else:
    raise ValueError("Error in janoschek, delta not legal")

## ilog2 
popt, pcov = leastsq(ilog2,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf),(1,np.inf)))
popts["ilog2"] = popt
convergency["ilog2"] = popt[0]

## exp3
popt, pcov = leastsq(exp3,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
if popt[1]<=0:
    raise ValueError("Error in exp3, a not legal")
popts["exp3"] = popt
convergency["exp3"] = popt[0]


## exp4
popt, pcov = leastsq(exp4,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf,-np.inf),(1,np.inf,np.inf,np.inf)))
if popt[1]<=0:
    raise ValueError("Error in exp4, a not legal")
if popt[3]<=0:
    raise ValueError("Error in exp4, alpha not legal")
popts["exp4"] = popt
convergency["exp4"] = popt[0]

## dr_hill_zero_background
popt, pcov = leastsq(dr_hill_zero_background,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
popts["dr_hill"] = popt
convergency["dr_hill"] = popt[0]

# then use leastsq get w
print(popts)
print()
print(convergency)

# then get the convergency point of each place

# then get the final ending point

# then get final convergency