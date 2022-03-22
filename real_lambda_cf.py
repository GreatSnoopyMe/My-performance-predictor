from asyncore import read
import numpy as np
from scipy.optimize import curve_fit, leastsq
from torch import norm, normal
from curve_functions import pow3, pow4, log_power, weibull, mmf, janoschek, ilog2, exp3, exp4, dr_hill_zero_background



all_curve_func = [pow3, pow4, log_power, weibull, mmf, janoschek, ilog2, exp3, exp4, dr_hill_zero_background]


# first get all the x,y
realresult = np.loadtxt('learning_curve1_result.txt')
y_train = np.loadtxt('learning_curve1.txt')
endingpoint = 199

N = y_train.shape[0]
print("Testing set length: ", realresult.shape[0])
print("Training set length: ", N)
x_train = np.arange(y_train.shape[0])+1

# first get all the theta
lowerbound = np.max(y_train)
popts = {}
convergency = []
expectedvalue = []
# implementation
y_getting = np.zeros((10,N))
## pow3
popt, pcov = curve_fit(pow3,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
popts["pow3"] = popt
convergency.append(popt[0])
expectedvalue.append(pow3(endingpoint,*popt))
y_getting[0] = pow3(x_train,*popt)
## pow4
popt, pcov = curve_fit(pow4,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf,-np.inf),(1,np.inf,np.inf,np.inf)))
popts["pow4"] = popt
convergency.append(popt[0])
expectedvalue.append(pow4(endingpoint,*popt))
y_getting[1] = pow4(x_train,*popt)

## log_power
popt, pcov = curve_fit(log_power,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
popts["log_power"] = popt
convergency.append(popt[0])
expectedvalue.append(log_power(endingpoint,*popt))
y_getting[2] = log_power(x_train,*popt)

## weibull
popt, pcov = curve_fit(weibull,xdata=x_train,ydata=y_train,bounds=((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)))
popts["weibull"] = popt
if popt[2]<0:
    raise ValueError("Error in weibull, kappa not legal")
if(popt[3])>=1:
    convergency.append(popt[0])
elif(popt[3]<1 and popt[3]>
0):
    convergency.append( popt[1])
else:
    raise ValueError("Error in weibull, delta not legal")
expectedvalue.append(weibull(endingpoint,*popt))
y_getting[3] = weibull(x_train,*popt)

## mmf
popt, pcov = curve_fit(mmf,xdata=x_train,ydata=y_train,bounds=((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)))
popts["mmf"] = popt
if popt[2]<0:
    raise ValueError("Error in mmf, kappa not legal")
if(popt[3])>=1:
    convergency.append(popt[0])
elif(popt[3]<1 and popt[3]>0):
    convergency.append(popt[1])
else:
    raise ValueError("Error in mmf, delta not legal")
expectedvalue.append(mmf(endingpoint,*popt))
y_getting[4] = mmf(x_train,*popt)

## janoschek
popt, pcov = curve_fit(janoschek,xdata=x_train,ydata=y_train,bounds=((0,0,-np.inf,-np.inf),(1,1,np.inf,np.inf)))
popts["janoschek"] = popt
if popt[2]<0:
    raise ValueError("Error in janoschek, kappa not legal")
if(popt[3])>=1:
    convergency.append(popt[0])
elif(popt[3]<1 and popt[3]>
0):
    convergency.append(popt[1])
else:
    raise ValueError("Error in janoschek, delta not legal")
expectedvalue.append(janoschek(endingpoint,*popt))
y_getting[5] = janoschek(x_train,*popt)

## ilog2 
popt, pcov = curve_fit(ilog2,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf),(1,np.inf)))
popts["ilog2"] = popt
convergency.append(popt[0])
expectedvalue.append(ilog2(endingpoint,*popt))
y_getting[6] = ilog2(x_train,*popt)

## exp3
popt, pcov = curve_fit(exp3,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
if popt[1]<=0:
    raise ValueError("Error in exp3, a not legal")
popts["exp3"] = popt
convergency.append(popt[0])
expectedvalue.append(exp3(endingpoint,*popt))
y_getting[7] = exp3(x_train,*popt)

## exp4
popt, pcov = curve_fit(exp4,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf,-np.inf),(1,np.inf,np.inf,np.inf)))
if popt[1]<=0:
    raise ValueError("Error in exp4, a not legal")
if popt[3]<=0:
    raise ValueError("Error in exp4, alpha not legal")
popts["exp4"] = popt
convergency.append(popt[0])
expectedvalue.append(exp4(endingpoint,*popt))
y_getting[8] = exp4(x_train,*popt)

## dr_hill_zero_background
popt, pcov = curve_fit(dr_hill_zero_background,xdata=x_train,ydata=y_train,bounds=((lowerbound,-np.inf,-np.inf),(1,np.inf,np.inf)))
popts["dr_hill"] = popt
convergency.append(popt[0])
expectedvalue.append(dr_hill_zero_background(endingpoint,*popt))
y_getting[9] = dr_hill_zero_background(x_train,*popt)


# # then use leastsq get w
print(y_getting.shape)
A_Left = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        for k in range(N):
            A_Left[i][j] += y_getting[i,k]*y_getting[j,k]
B_right = np.zeros((10,1))
# for i in range(10):
    # B_right[i] = expectedvalue[i]

B_right = y_getting.dot(y_train[np.newaxis].T)

lambda_k = 7.00e-07


A_tmp = A_Left + np.eye(10) * lambda_k
weights = np.linalg.solve(A_tmp,B_right)
# normalized_weights = (weights-np.mean(weights))/np.std(weights)+0.1
# print("checking result:",np.sum(np.square(A_Left.dot(weights)-B_right)))
# then get the convergency point of each place
expected_convergency = 0
expected_ending = 0
for kk in range(10):
    expected_convergency += convergency[kk]*weights[kk]

for value_e in range(10):
    expected_ending += expectedvalue[value_e]*weights[value_e]

print("expected convergency to be: ", expected_convergency)
print("expected ending to be: ", expected_ending)
