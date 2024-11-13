# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from functions import CQModel
import matplotlib.cm as cm

# Step 1: Read dataset and Process data
def CQFitPlot():
    site = 212058
    Q_thre = 2
    date_postfire = pd.to_datetime('2020-01-17')
    fig_dir = f'../output/figs/{site}/'
    storm_data = pd.read_csv(f'../output/CQ_analysis/{site}/' + \
                                f'Q_above_{Q_thre}_{site}_StormEventRefilterData.csv', index_col = 'id')
    storm_data['Datetime'] = pd.to_datetime(storm_data['Datetime'])
    storm_data_pre = storm_data[storm_data['Datetime'] < date_postfire]
    index_date_post = storm_data[storm_data['Datetime'] >= date_postfire].index[0]
    storm_data_post = storm_data[storm_data['Datetime'] >= date_postfire]
    cols = ['Discharge (cms)', 'Turbidity (NTU)']
    mod_type = 'mixed'

    # Set CQModel class
    CQM = CQModel(mod_type, initial_params = [1, 1, 1, 1])
    # Step 2: Curve-fit to obtain parameter values
    pre_obs_conc = storm_data_pre.loc[:, cols[1]].values
    post_obs_conc = storm_data_post.loc[:, cols[1]].values
    if mod_type =='power_law':
        pre_obs_flow = storm_data_pre.loc[:, cols[0]].values
        post_obs_flow = storm_data_post.loc[:, cols[0]].values
        popt, pcov = CQM.fit(pre_obs_flow, pre_obs_conc)
        # Estimate NTU using fitted CQ model.
        # Plot 1: Time series
        storm_data_pre['Estimate_Turbidity'] = CQM.evaluate(pre_obs_flow, popt)
        storm_data_post['Estimate_Turbidity'] = CQM.evaluate(storm_data_post.loc[:, cols[0]].values, popt)
        storm_data.loc[:index_date_post, 'Est_Tbdt_power_precal'] = storm_data_pre['Estimate_Turbidity']
        storm_data.loc[index_date_post:, 'Est_Tbdt_power_precal'] = storm_data_post['Estimate_Turbidity']
    else: # Get total, quick and base flow
        pre_obs_flow = np.zeros(shape = (storm_data_pre.loc[:, cols[0]].values.shape[0], 3))
        post_obs_flow = np.zeros(shape = (storm_data_post.loc[:, cols[0]].values.shape[0], 3))
        pre_obs_flow[:, 0] = storm_data_pre.loc[:, 'total_flow'].values
        pre_obs_flow[:, 1] = storm_data_pre.loc[:, 'storm_flow'].values
        pre_obs_flow[:, 2] = storm_data_pre.loc[:, 'base_flow'].values
        post_obs_flow[:, 0] = storm_data_post.loc[:, 'total_flow'].values
        post_obs_flow[:, 1] = storm_data_post.loc[:, 'storm_flow'].values
        post_obs_flow[:, 2] = storm_data_post.loc[:, 'base_flow'].values
        # result = CQM.fit(pre_obs_flow, pre_obs_conc)
        result = CQM.fit(post_obs_flow, post_obs_conc)
        # Estimate NTU using fitted CQ model.
        storm_data_pre['Estimate_Turbidity'] = CQM.evaluate(pre_obs_flow, result.x)
        storm_data_post['Estimate_Turbidity'] = CQM.evaluate(post_obs_flow, result.x)
        storm_data.loc[:index_date_post, 'Est_Tbdt_mix_postcal'] = storm_data_pre['Estimate_Turbidity']
        storm_data.loc[index_date_post:, 'Est_Tbdt_mix_postcal'] = storm_data_post['Estimate_Turbidity']
    storm_data.to_csv(f'../output/CQ_analysis/{site}/' + \
                        f'Q_above_{Q_thre}_{site}_StormEventRefilterData.csv')
    # Calculate R2

    lab_fs = 12
    tick_fs = 11
    # Plot 2: Scatter plot of turbidity
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(14, 5))
    axes[0].scatter(pre_obs_flow[:, 0], pre_obs_conc, s = 2, color = 'grey', label = 'Prefire', alpha=0.3)
    axes[0].scatter(pre_obs_flow[:, 0], storm_data_pre['Estimate_Turbidity'].values, s = 2,
            color = 'orange', alpha = 0.8) # label='fit: a=%5.3f, b=%5.3f' % tuple(popt), 

    # Using gradient colors for data one year before fire
    time_sel = storm_data_pre[(storm_data_pre.Datetime >= pd.to_datetime('2018-07-01'))]
    date_nums = time_sel.Datetime.astype(np.int64) / 1e9
    norm = plt.Normalize(date_nums.min(), date_nums.max())
    colors = cm.viridis(norm(date_nums)) 
    scatter_pre = axes[0].scatter(time_sel[cols[0]], time_sel[cols[1]], s = 2, color = colors, label = '2018-07--2020-01')
    #Add color bars
    cbar = plt.colorbar(scatter_pre)
    # Settings for the figure
    axes[0].legend(fontsize = 12)
    axes[0].tick_params(axis='both', which = 'major', labelsize = tick_fs)
    axes[0].set_ylabel('Turbidity (NTU)', fontsize = lab_fs)
    axes[0].set_xlabel('Flow (cms)', fontsize = lab_fs)
    axes[1].scatter(post_obs_flow[:, 0], post_obs_conc, s = 2,
            label='Postfire', color = 'grey', alpha=0.3)
    axes[1].scatter(post_obs_flow[:, 0], storm_data_post['Estimate_Turbidity'].values, s = 2,
            color = 'orange', alpha = 0.8) # label='fit: a=%5.3f, b=%5.3f' % tuple(popt), 

    # Using gradient colors for data one year after fire
    time_post = storm_data_post[(storm_data_post.Datetime < pd.to_datetime('2021-06-30'))]
    date_nums_post = time_post.Datetime.astype(np.int64) / 1e9
    norm = plt.Normalize(date_nums_post.min(), date_nums_post.max())
    colors = cm.viridis(norm(date_nums_post)) 
    scatter_post = axes[1].scatter(time_post[cols[0]], time_post[cols[1]], s = 2, color = colors, label = '2020-01--2021-06')
    #Add color bars
    cbar = plt.colorbar(scatter_post)
    axes[1].legend(fontsize = lab_fs)
    axes[1].set_xlabel('Flow (cms)', fontsize = lab_fs)
    axes[1].tick_params(axis='both', which = 'major', labelsize = tick_fs)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(f'{fig_dir}{mod_type}PostcalCQFitPrePostfire.png', format='png', dpi=300)


    # Plot the residuals
    # Plot 2: Scatter plot of turbidity
    pre_res = storm_data_pre['Estimate_Turbidity'].values - pre_obs_conc
    post_res = storm_data_post['Estimate_Turbidity'].values - post_obs_conc
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(12, 5))
    axes[0].scatter(pre_obs_flow[:, 0], pre_res, s = 2, color = 'blue', label = 'Prefire')
    axes[0].legend(fontsize = lab_fs)
    axes[0].tick_params(axis='both', which = 'major', labelsize = tick_fs)
    axes[0].set_ylabel('Turbidity Residuals (NTU)', fontsize = lab_fs)
    axes[0].set_xlabel('Flow (cms)', fontsize = lab_fs)
    axes[1].scatter(post_obs_flow[:, 0], post_res, s = 2,
            label='Postfire', color = 'orange')
    axes[1].legend(fontsize = lab_fs)
    axes[1].set_xlabel('Flow (cms)', fontsize = lab_fs)
    axes[1].tick_params(axis='both', which = 'major', labelsize = tick_fs)
    # plt.yscale('log')
    plt.title('Model - Observation', fontsize = lab_fs)
    plt.xscale('log')
    plt.savefig(f'{fig_dir}{mod_type}PostcalResidualCQFitPrePostfire.png', format='png', dpi=300)

    fig = plt.figure(figsize=(8, 5))
    data_temp = [pre_obs_conc, post_obs_conc, time_post.dropna(how='any')[cols[1]]] #, time_post.dropna(how='any')[cols[1]]
    labels = ['All Data Prefire', 'All Data Postfire', '2020-01--2021-06']#, '1 year after fire']
    sns.boxplot(data_temp)
    plt.xticks(ticks=[0, 1, 2], labels=labels)
    plt.yscale('log')
    plt.savefig(f'{fig_dir}BoxplotConcDifferentPeriod.png', format='png', dpi=300)

if __name__ == '__main__':
    CQFitPlot()