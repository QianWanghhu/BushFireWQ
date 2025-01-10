# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from functions import CQModel
import matplotlib.cm as cm
from spotpy.objectivefunctions import nashsutcliffe, pbias, rrmse

# Step 1: Read dataset and Process data
def CQFitPlot(mod_type, train_period, fig_dir):
    site = 212058
    Q_thre = 1
    start_postyear = pd.to_datetime('2020-01-01')
    end_postyear = pd.to_datetime('2021-07-01')
    start_preyear = pd.to_datetime('2016-07-01')
    storm_data = pd.read_csv(f'../output/CQ_analysis/{site}/' + \
                            f'Q_above_{Q_thre}_{site}_StormEventRefilterData.csv', index_col='id') # '212058_NTU_Discharge_2016_2021.csv'
    storm_data['Datetime'] = pd.to_datetime(storm_data['Datetime'], format='mixed', dayfirst=True)
    if train_period == 'Full': # If using the full dataset for training CQ model, set the end date as end_postfire.
        pre_time_tf = (storm_data['Datetime'] <= end_postyear) & (storm_data['Datetime'] >= start_preyear)
    else: # If using the prefire dataset for training CQ model, set the end date as start_postyear.
        pre_time_tf = (storm_data['Datetime'] < start_postyear) & (storm_data['Datetime'] >= start_preyear)
    post_time_tf = (storm_data['Datetime'] >= start_postyear) & (storm_data['Datetime'] < end_postyear)
    storm_data_pre = storm_data[pre_time_tf]
    index_date_post = storm_data[storm_data['Datetime'] >= start_postyear].index[0]
    storm_data_post = storm_data[post_time_tf]

    cols = ['Discharge (cms)', 'Turbidity (NTU)']
    # Set CQModel class
    CQM = CQModel(mod_type, initial_params = [1, 1, 1, 1])
    # Step 2: Curve-fit to obtain parameter values
    pre_obs_conc = storm_data_pre.loc[:, cols[1]].values
    post_obs_conc = storm_data_post.loc[:, cols[1]].values
    # Calculate NSE for two periods
    nse = {'Prefire': 0, 'Postfire':0}
    mec_rrmse = {'Prefire': 0, 'Postfire':0}
    mec_pbias = {'Prefire': 0, 'Postfire':0}
    if mod_type =='power_law':
        pre_obs_flow = storm_data_pre.loc[:, cols[0]].values
        post_obs_flow = storm_data_post.loc[:, cols[0]].values
        result_fit, _ = CQM.fit(pre_obs_flow, pre_obs_conc)
        # Estimate NTU using fitted CQ model.
        # Plot 1: Time series
        storm_data_pre['Estimate_Turbidity'] = CQM.evaluate(pre_obs_flow, result_fit)
        storm_data_post['Estimate_Turbidity'] = CQM.evaluate(post_obs_flow, result_fit)
        storm_data.loc[storm_data[pre_time_tf].index, 'Est_Tbdt_power_precal'] = storm_data_pre['Estimate_Turbidity'].values
        storm_data.loc[storm_data[post_time_tf].index, 'Est_Tbdt_power_precal'] = storm_data_post['Estimate_Turbidity'].values
    else: # Get total, quick and base flow
        storm_data.loc[:, f'Est_Tbdt_mix_{train_period}cal'] = None
        pre_obs_flow = np.zeros(shape = (storm_data_pre.loc[:, cols[0]].values.shape[0], 3))
        post_obs_flow = np.zeros(shape = (storm_data_post.loc[:, cols[0]].values.shape[0], 3))
        pre_obs_flow[:, 0] = storm_data_pre.loc[:, 'total_flow'].values
        pre_obs_flow[:, 1] = storm_data_pre.loc[:, 'storm_flow'].values
        pre_obs_flow[:, 2] = storm_data_pre.loc[:, 'base_flow'].values
        post_obs_flow[:, 0] = storm_data_post.loc[:, 'total_flow'].values
        post_obs_flow[:, 1] = storm_data_post.loc[:, 'storm_flow'].values
        post_obs_flow[:, 2] = storm_data_post.loc[:, 'base_flow'].values
        # Train CQ model with Prefire events if train_period == 'Pre', otherwise using Postfire events
        if train_period == 'Pre':
            result_fit = CQM.fit(pre_obs_flow, pre_obs_conc)
        elif train_period == 'Post':
            result_fit = CQM.fit(post_obs_flow, post_obs_conc)
        else:
            result_fit = CQM.fit(pre_obs_flow, pre_obs_conc)
        # Check if the optimization was successful
        # if result_fit.success:
        #     print("Optimization failed:", result_fit.message)

        # Estimate NTU using fitted CQ model.
        storm_data_pre['Estimate_Turbidity'] = CQM.evaluate(pre_obs_flow, result_fit['x'])
        storm_data_post['Estimate_Turbidity'] = CQM.evaluate(post_obs_flow, result_fit['x'])
        storm_data.loc[storm_data[pre_time_tf].index, f'Est_Tbdt_mix_{train_period}cal'] = \
            storm_data_pre['Estimate_Turbidity'].values
        storm_data.loc[storm_data[post_time_tf].index, f'Est_Tbdt_mix_{train_period}cal'] = \
            storm_data_post['Estimate_Turbidity'].values
    storm_data.to_csv(f'../output/CQ_analysis/{site}/' + \
                    f'Q_above_{Q_thre}_{site}_StormEventRefilterData.csv') #'212058_NTU_Discharge_2016_2021.csv'
    # Calculate performance metrics
    # Calculate the nse with only storm data
    storm_data_sel = pd.read_csv(f'../output/CQ_analysis/{site}/' + \
                                f'Q_above_{Q_thre}_{site}_StormEventRefilterData.csv', index_col='id')#
    storm_data_sel['Datetime'] = pd.to_datetime(storm_data_sel['Datetime'], format='mixed', dayfirst=True)
    filtered_df_pre = storm_data_pre.merge(storm_data_sel, on='Datetime')
    # Filter the data according to datetime in storm_data_sel
    # breakpoint()
    nse['Prefire'] = np.round(nashsutcliffe(filtered_df_pre['Turbidity (NTU)_x'], \
                                            filtered_df_pre['Estimate_Turbidity']), 3)
    mec_rrmse['Prefire'] = np.round(rrmse(filtered_df_pre['Turbidity (NTU)_x'], \
                                            filtered_df_pre['Estimate_Turbidity']), 3)
    mec_pbias['Prefire'] = np.round(pbias(filtered_df_pre['Turbidity (NTU)_x'], \
                                            filtered_df_pre['Estimate_Turbidity']), 3)
    # nse['Postfire'] = np.round(nashsutcliffe(storm_data_post['Turbidity (NTU)'], storm_data_post['Estimate_Turbidity']), 3)
    filtered_df_post = storm_data_post.merge(storm_data_sel, on='Datetime')
    nse['Postfire'] = np.round(nashsutcliffe(filtered_df_post['Turbidity (NTU)_x'], \
                                            filtered_df_post['Estimate_Turbidity']), 3)
    mec_rrmse['Postfire'] = np.round(rrmse(filtered_df_post['Turbidity (NTU)_x'], \
                                            filtered_df_post['Estimate_Turbidity']), 3)
    mec_pbias['Postfire'] = np.round(pbias(filtered_df_post['Turbidity (NTU)_x'], \
                                            filtered_df_post['Estimate_Turbidity']), 3)

    # Set fontsize used in plots
    lab_fs = 14
    tick_fs = 14
    # Plot 2: Scatter plot of turbidity
    _, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True, figsize=(9, 6))
    # Using gradient colors for data one year before fire
    date_nums = storm_data_pre.Datetime.astype(np.int64) / 1e9
    norm = plt.Normalize(date_nums.min(), date_nums.max())
    colors = cm.viridis(norm(date_nums)) 
    if mod_type == 'power_law':
        scatter_pre = axes[0, 0].scatter(storm_data_pre[cols[0]], storm_data_pre[cols[1]], s = 2, color = colors, label = 'Prefire')
        axes[0, 1].scatter(pre_obs_flow[:], storm_data_pre['Estimate_Turbidity'].values, s = 2,
                color = 'orange', alpha = 0.8)       
    else:
        scatter_pre = axes[0, 0].scatter(pre_obs_flow[:, 0], pre_obs_conc, s = 2, color = colors, label = 'Prefire')
        axes[0, 1].scatter(pre_obs_flow[:, 0], storm_data_pre['Estimate_Turbidity'].values, s = 2,
                label='CQ model', color = 'orange', alpha = 0.8) # label='fit: a=%5.3f, b=%5.3f' % tuple(popt), 
    #Add color bars
    axes[0, 0].set_title('Observation', fontsize = lab_fs)
    axes[0, 1].set_title('CQ model', fontsize = lab_fs)
    axes[0, 1].tick_params(axis='both', which = 'major', labelsize = tick_fs)
    axes[0, 1].text(4, 0.9, f'NSE={nse["Prefire"]}', fontsize=12, ha='center', va='center', color='black')
    cbar = plt.colorbar(scatter_pre)
    cbar.set_ticks([0, 1], labels=[(str(storm_data_pre.Datetime.values[0])[:10]), str(storm_data_pre.Datetime.values[-1])[:10]])
    
    # Settings for the figure
    axes[0, 0].legend(fontsize = lab_fs)
    axes[0, 0].tick_params(axis='both', which = 'major', labelsize = tick_fs)
    axes[0, 0].set_ylabel('Turbidity (NTU)', fontsize = lab_fs)
    axes[1, 0].set_ylabel('Turbidity (NTU)', fontsize = lab_fs)
    axes[0, 0].set_xlabel('Flow (cms)', fontsize = lab_fs)

    date_nums_post = storm_data_post.Datetime.astype(np.int64) / 1e9
    norm_post = plt.Normalize(date_nums_post.min(), date_nums_post.max())
    colors_post = cm.viridis(norm_post(date_nums_post)) 
    if mod_type == 'power_law':
        scatter_post = axes[1, 0].scatter(post_obs_flow[:], post_obs_conc, s = 2,
            label='Postfire', color = colors_post, alpha=0.8)
        axes[1, 1].scatter(post_obs_flow[:], storm_data_post['Estimate_Turbidity'].values, s = 2,
            color = 'orange', alpha = 0.8) #  
    else:
        scatter_post = axes[1, 0].scatter(post_obs_flow[:, 0], post_obs_conc, s = 2,
            label='Postfire', color = colors_post, alpha=0.8)
        axes[1, 1].scatter(post_obs_flow[:, 0], storm_data_post['Estimate_Turbidity'].values, s = 2,
            label = 'CQ model', color = 'orange', alpha = 0.8) # label='fit: a=%5.3f, b=%5.3f' % tuple(popt), 
    cbar_post = plt.colorbar(scatter_post)
    cbar_post.set_ticks([0, 1], labels=[str(storm_data_post.Datetime.values[0])[:10], str(storm_data_post.Datetime.values[-1])[:10]])
    # Set xlabel and tick labels
    axes[1, 0].legend(fontsize = lab_fs)
    axes[1, 0].set_xlabel('Flow (cms)', fontsize = lab_fs)
    axes[1, 0].tick_params(axis='both', which = 'major', labelsize = tick_fs)
    axes[1, 1].tick_params(axis='both', which = 'major', labelsize = tick_fs)
    axes[1, 1].text(4, 0.9, f'NSE = {nse["Postfire"]}', fontsize=12, ha='center', va='center', color='black')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(f'{fig_dir}{mod_type}{train_period}calCQFitPrePostfire.png', format='png', dpi=300)

    # Plot the residuals
    # Plot 2: Scatter plot of turbidity
    # Calculate the ratio of mod / obs
    pre_res = storm_data_pre['Estimate_Turbidity'].values / pre_obs_conc
    post_res = storm_data_post['Estimate_Turbidity'].values / post_obs_conc
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(6, 3))
    if mod_type == 'power_law':
        axes[0].scatter(pre_obs_flow[:], pre_res, s = 2, color = 'blue', label = 'Prefire')  
    else:
        axes[0].scatter(pre_obs_flow[:, 0], pre_res, s = 2, color = 'blue', label = 'Prefire')  
    axes[0].legend(fontsize = lab_fs)
    axes[0].tick_params(axis='both', which = 'major', labelsize = tick_fs)
    axes[0].set_ylabel('Turbidity Residuals (NTU)', fontsize = lab_fs)
    axes[0].set_xlabel('Flow (cms)', fontsize = lab_fs)
    if mod_type == 'power_law':
        axes[1].scatter(post_obs_flow[:], post_res, s = 2,
            label='Postfire', color = 'orange')
    else:
        axes[1].scatter(post_obs_flow[:, 0], post_res, s = 2,
            label='Postfire', color = 'orange')
    axes[1].legend(fontsize = lab_fs)
    axes[1].set_xlabel('Flow (cms)', fontsize = lab_fs)
    axes[1].tick_params(axis='both', which = 'major', labelsize = tick_fs)
    plt.yscale('log')
    plt.suptitle('Model / Observation', fontsize = lab_fs, y=0.95)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}{mod_type}{train_period}calResidualCQFit.png', format='png', dpi=300)

    fig = plt.figure(figsize=(8, 5))
    data_temp = [pre_obs_conc, post_obs_conc] #
    labels = ['Data Prefire', 'Data Postfire']#
    plt.boxplot(data_temp)
    plt.xticks(ticks=[0, 1], labels=labels)
    plt.yscale('log')
    plt.savefig(f'{fig_dir}BoxplotConcDifferentPeriod.png', format='png', dpi=300)
    return {'result_fit':result_fit, 'nse': nse, 'rrmse':mec_rrmse, 'pbias':mec_pbias}

if __name__ == '__main__':
    site = 212058
    fig_dir = f'../output/figs/{site}/CQ_LM/'
    mod_types = ['power_law', 'mixed']
    train_periods = ['Pre', 'Post', 'Full']
    # Define the index for coeff with the first being CQ model coefficients and the remaining for NSE and log NSE
    keys = ['aq', 'bq', 'aq_std', 'bq_std', 'nse_pre', 'nse_post', 'pbias_pre', 'pbias_post', 'rrmse_pre', 'rrmse_post']
    coeff = pd.DataFrame(index=keys, columns=train_periods)
    _ = CQFitPlot(mod_types[0], train_periods[0], fig_dir)
    temp = CQFitPlot(mod_types[1], train_periods[0], fig_dir)
    temp2 = CQFitPlot(mod_types[1], train_periods[1], fig_dir)
    temp3 = CQFitPlot(mod_types[1], train_periods[2], fig_dir)
    for idx, res in enumerate([temp, temp2, temp3]):
        std_devs = np.sqrt(np.diag(res['result_fit']['std']))
        coeff.loc[keys[0:2], train_periods[idx]] = res['result_fit']['x']
        coeff.loc[keys[2:4], train_periods[idx]] = np.diagonal(std_devs)
        coeff.loc[keys[4], train_periods[idx]] = res['nse']['Prefire']
        coeff.loc[keys[5], train_periods[idx]] = res['nse']['Postfire']
        coeff.loc[keys[6], train_periods[idx]] = res['pbias']['Prefire']
        coeff.loc[keys[7], train_periods[idx]] = res['pbias']['Postfire']
        coeff.loc[keys[8], train_periods[idx]] = res['rrmse']['Prefire']
        coeff.loc[keys[9], train_periods[idx]] = res['rrmse']['Postfire']
    coeff = coeff.astype(float)
    coeff.to_csv(f'{fig_dir}CQModelCoeff_LM_storm_total.csv')