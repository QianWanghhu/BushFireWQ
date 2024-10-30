# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from functions import CQModel

# Step 1: Read dataset and Process data
site = 212058
Q_thre = 2
date_postfire = pd.to_datetime('2020-01-17')
fig_dir = f'../output/figs/{site}/'
storm_data = pd.read_csv(f'../output/CQ_analysis/{site}/' + \
                            f'Q_above_{Q_thre}_{site}_StormEventRefilterData.csv', index_col = 'id')
storm_data['Datetime'] = pd.to_datetime(storm_data['Datetime'])
storm_data_pre = storm_data[storm_data['Datetime'] < date_postfire]
storm_data_post = storm_data[storm_data['Datetime'] >= date_postfire]
cols = ['Discharge (cms)', 'Turbidity (NTU)']

# Set CQModel class
CQM = CQModel()
# Step 2: Curve-fit to obtain parameter values
pre_obs_flow = storm_data_pre.loc[:, cols[0]].values
pre_obs_conc = storm_data_pre.loc[:, cols[1]].values
post_obs_flow = storm_data_post.loc[:, cols[0]].values
post_obs_conc = storm_data_post.loc[:, cols[1]].values
popt, pcov = CQM.fit(pre_obs_flow, pre_obs_conc)

# Estimate NTU using fitted CQ model.
# Plot 1: Time series
storm_data_pre['Estimate_Turbidity'] = CQM.evaluate(pre_obs_flow, popt)
storm_data_post['Estimate_Turbidity'] = CQM.evaluate(storm_data_post.loc[:, cols[0]].values, popt)
# Calculate R2


# Plot 2: Scatter plot of turbidity
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(12, 5))
axes[0].scatter(pre_obs_flow, pre_obs_conc, s = 2, color = 'blue', label = 'Prefire')
axes[0].scatter(pre_obs_flow, storm_data_pre['Estimate_Turbidity'].values, s = 2,
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt), color = 'purple')
axes[0].legend(fontsize = 12)
axes[0].tick_params(axis='both', which = 'major', labelsize = 11)
axes[0].set_ylabel('Turbidity (NTU)', fontsize = 12)
axes[0].set_xlabel('Flow (cms)', fontsize = 12)
axes[1].scatter(post_obs_flow, post_obs_conc, s = 2,
         label='Postfire', color = 'orange')
axes[1].scatter(post_obs_flow, storm_data_post['Estimate_Turbidity'].values, s = 2,
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt), color = 'purple')
axes[1].legend(fontsize = 12)
axes[1].set_xlabel('Flow (cms)', fontsize = 12)
axes[1].tick_params(axis='both', which = 'major', labelsize = 11)
plt.yscale('log')
plt.xscale('log')
plt.savefig(f'{fig_dir}CQPowerLawFitPrePostfire.png', format='png', dpi=300)


# Plot the residuals
# Plot 2: Scatter plot of turbidity
pre_res = storm_data_pre['Estimate_Turbidity'].values - pre_obs_conc
post_res = storm_data_post['Estimate_Turbidity'].values - post_obs_conc
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(12, 5))
axes[0].scatter(pre_obs_flow, pre_res, s = 2, color = 'blue', label = 'Prefire')
axes[0].legend(fontsize = 12)
axes[0].tick_params(axis='both', which = 'major', labelsize = 11)
axes[0].set_ylabel('Turbidity Residuals (NTU)', fontsize = 12)
axes[0].set_xlabel('Flow (cms)', fontsize = 12)
axes[1].scatter(storm_data_post.loc[:, cols[0]].values, post_res, s = 2,
         label='Postfire', color = 'orange')
axes[1].legend(fontsize = 12)
axes[1].set_xlabel('Flow (cms)', fontsize = 12)
axes[1].tick_params(axis='both', which = 'major', labelsize = 11)
plt.yscale('log')
plt.xscale('log')
plt.savefig(f'{fig_dir}ResidualCQPowerLawFitPrePostfire.png', format='png', dpi=300)