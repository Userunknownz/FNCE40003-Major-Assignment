# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vkmiPWC4GMCl28ZnynHlZeyO91Dl-Q7t
"""

# Original Author   : John Tsang
# Modified by: Hanqing Tian, Haixuan Chen
# Date     : December 7th, 2017
# Purpose  : Implement the Diebold-Mariano Test (DM test) to compare 
#            forecast accuracy
# Input    : 1) actual_lst: the list of actual values
#            2) pred1_lst : the first list of predicted values
#            3) pred2_lst : the second list of predicted values
#            4) h         : the number of stpes ahead
#            5) crit      : a string specifying the criterion 
#                             i)  MSE : the mean squared error
#                            ii)  MAD : the mean absolute deviation
#                           iii) MAPE : the mean absolute percentage error
#                            iv) poly : use power function to weigh the errors
#            6) poly      : the power for crit power 
#                           (it is only meaningful when crit is "poly")
# Condition: 1) length of actual_lst, pred1_lst and pred2_lst is equal
#            2) h must be an integer and it must be greater than 0 and less than 
#               the length of actual_lst.
#            3) crit must take the 4 values specified in Input
#            4) Each value of actual_lst, pred1_lst and pred2_lst must
#               be numerical values. Missing values will not be accepted.
#            5) power must be a numerical value.
# Return   : a named-tuple of 2 elements
#            1) p_value : the p-value of the DM test
#            2) DM      : the test statistics of the DM test
##########################################################
# References:
#
# Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of 
#   prediction mean squared errors. International Journal of forecasting, 
#   13(2), 281-291.
#
# Diebold, F. X. and Mariano, R. S. (1995), Comparing predictive accuracy, 
#   Journal of business & economic statistics 13(3), 253-264.
#
##########################################################
def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    elif (crit == "QLIKE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((np.log(p1) + (actual / p1)))
            e2_lst.append((np.log(p2) + (actual / p2)))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)
    
    return rt

df = pd.read_csv('R_Inputs.csv')
df.head()

df = df[df['HAR'].isnull() == False ]

df['d'] = pd.to_datetime(df.Date)

display(dm_test(df['ACTUAL'], df['HAR'], df['HAR-ALL'], h = 1, crit="QLIKE"))
display(dm_test(df['ACTUAL'], df['HAR'], df['HAR-ALL'], h = 1, crit="MSE"))
display(dm_test(df['ACTUAL'], df['HAR-RSV'], df['HAR-RSV-ALL'], h = 1, crit="QLIKE"))
display(dm_test(df['ACTUAL'], df['HAR-RSV'], df['HAR-RSV-ALL'], h = 1, crit="MSE"))
display(dm_test(df['ACTUAL'], df['HAR-ALL'], df['HAR-RSV-ALL'], h = 1, crit="QLIKE"))
display(dm_test(df['ACTUAL'], df['HAR-ALL'], df['HAR-RSV-ALL'], h = 1, crit="MSE"))

df1 = df.copy()
df1 = df1[df1.index > '2020-01-01' ]

display(dm_test(df1['ACTUAL'], df1['HAR-RSV'], df1['HAR-RSV-ALL'], h = 1, crit="QLIKE"))

display(dm_test(df1['ACTUAL'], df1['HAR-ALL'], df1['HAR-RSV-ALL'], h = 1, crit="QLIKE"))

display(dm_test(df1['ACTUAL'], df1['HAR'], df1['HAR-ALL'], h = 1, crit="QLIKE"))
display(dm_test(df1['ACTUAL'], df1['HAR'], df1['HAR-ALL'], h = 1, crit="MSE"))
display(dm_test(df1['ACTUAL'], df1['HAR-RSV'], df1['HAR-RSV-ALL'], h = 1, crit="QLIKE"))
display(dm_test(df1['ACTUAL'], df1['HAR-RSV'], df1['HAR-RSV-ALL'], h = 1, crit="MSE"))
display(dm_test(df1['ACTUAL'], df1['HAR'], df1['HAR-RSV'], h = 1, crit="QLIKE"))
display(dm_test(df1['ACTUAL'], df1['HAR'], df1['HAR-RSV'], h = 1, crit="MSE"))
display(dm_test(df1['ACTUAL'], df1['HAR-ALL'], df1['HAR-RSV-ALL'], h = 1, crit="QLIKE"))
display(dm_test(df1['ACTUAL'], df1['HAR-ALL'], df1['HAR-RSV-ALL'], h = 1, crit="MSE"))

df2 = df[df.index > '2020-01-01']

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
#df.set_index('d',inplace=True)
df2['Model_Average'] = 0.5 * df2['HAR-ALL'] + 0.5 * df2['HAR-RSV-ALL']
df2[['ACTUAL','Model_Average']].plot(figsize=(14,7))
plt.xlabel('Time')
plt.ylabel('Realized Volatility')
plt.title('Model Average Prediction and the Actual oil volatility (2020)')
plt.savefig('2020.jpg')

plt.figure(figsize=(14,7))
plt.hist(df.ACTUAL,bins=100)
plt.title('Actual Realized Volatility',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.savefig('histogram_of rv.PNG')

df.ACTUAL.skew()