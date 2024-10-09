# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sci_opt
import datetime

# Read dataset
fn = '../data/212058_Daily.csv'
cols = ['Discharge (ML/d)', 'Turbidity (NTU)']
cq_data = pd.read_csv(fn, index_col = 'Date and time', skiprows = 3, usecols = [0, 1, 3])
cq_data.rename(columns={cq_data.columns[ii]: cols[ii] for ii in range(len(cols))}, inplace = True)
cq_data.index.name = 'Datetime'
# Process data
# Drop rows if Turbidity is of value NaN
cq_data.dropna(axis = 0, how = 'any', inplace = True)
cq_data.index = pd.to_datetime(cq_data.index, dayfirst = True)
# Fit a C-Q model in power-law relationship
# Step 1: Define a power-law function
def func(x, a, b):
    return a * np.power(x, b)

# Step 2: Curve-fit to obtain parameter values
obs_flow = cq_data.loc[:, cols[0]].values
obs_conc = cq_data.loc[:, cols[1]].values
popt, pcov = sci_opt.curve_fit(func, xdata = obs_flow,  ydata = cq_data.loc[:, cols[1]].values)
estimate_conc = func(obs_flow, *popt)

# Plot 1: Time series
cq_data['Estimate_Turbidity'] = estimate_conc
sns.lineplot(data = cq_data, x = cols[0], y = cols[1])
sns.lineplot(data = cq_data, x = cols[0], y = cq_data.columns[-1])

# Plot 2: Scatter plot of turbidity
ax = plt.scatter(obs_conc, estimate_conc, s = 2,
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.yscale('log')
plt.xscale('log')

# Plot 3: C-Q plot

# Plot 4: Hysteresis plot