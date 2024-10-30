import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sci_opt

# This file contains functions for data analysis
def EventFilter(data, event_info, Qthresh, q_name, time_freq):
    """Storm events are further filtered based on the flow peak. The original events are identified by R packages."""
    # Loop over events to get peak flow. 
    for ii in range(event_info.shape[0]):
        if time_freq == 'D':
            start_time, end_time = pd.Timestamp(event_info.start[ii].date()), pd.Timestamp(event_info.end[ii].date())
            # Filter data according to start and end time.
            filtered_df = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)]
            if start_time == end_time:
                # if ii == 287:
                #     breakpoint()
                event_info.loc[ii, 'q_peak'] = filtered_df[q_name].values
            else:
                event_info.loc[ii, 'q_peak'] = filtered_df[q_name].max()
        elif time_freq == 'H':
            start_time, end_time = event_info.start[ii], event_info.end[ii]
            # Filter data according to start and end time.
            filtered_df = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)]
            event_info.loc[ii, 'q_peak'] = filtered_df[q_name].max()
        else:
            print('The temporal frequency is not supported.')
    event_info[f'Event_filter_{Qthresh}'] = np.where(event_info['q_peak'] >= Qthresh, 1, 0)
    return event_info  

def EventDataComb(data, event_info, Qthresh, time_freq):
    """Function for Combine all C and Q data for given events. 
    -- Return Dataframe with Datetime as index; flow as the first column and conc as the second column.
        StormEventID as the third column.
    """
    storm_df = pd.DataFrame(data = None, columns = data.columns)
    storm_df['stormID'] = None
    storm_df['stormCount'] = None
    k_count = 0
    for ii in range(event_info.shape[0]):
        if event_info.loc[ii, f'Event_filter_{Qthresh}']:
            k_count += 1
            # TODO Setup code for using daily data.
            if time_freq == 'D':
                start_time, end_time = pd.Timestamp(event_info.start[ii].date()), pd.Timestamp(event_info.end[ii].date())
            elif time_freq =='H':
                start_time, end_time = event_info.start[ii], event_info.end[ii]
            else:
                print('The temporal frequency is not identified. Only D or H are accepted')
                break
            filtered_df = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)]
            filtered_df.loc[:, 'stormID'] = ii
            filtered_df.loc[:, 'stormCount'] = k_count
            storm_df = pd.concat([storm_df, filtered_df], axis=0)

    return storm_df

class CQModel:
    # Fit a C-Q model in power-law relationship
    # Step 1: Define a power-law function
    def __init__(self):
        pass

    def func(self, x, a, b):
        """
        Define the function type.
        """
        return a * np.power(x, b)

    # Step 2: Curve-fit to obtain parameter values
    def fit(self, flow, conc):
        """
        Fit the function using given flow and concentration observations.
        """
        popt, pcov = sci_opt.curve_fit(self.func, xdata = flow, ydata = conc)
        return popt, pcov
    
    def evaluate(self, flow, popt):
        """
        Calculate estimated concentrations using the fitted function.
        """
        estimate_conc = self.func(flow, *popt)
        return estimate_conc

# Create scatter plot
def plot_storm_cq(storm_data, x_lab, y_lab, freq, site, date_postfire, Q_thr, colors = ['blue', 'orange'], marker='o', \
                  alpha = 0.5, labels = ['Prefire', 'Postfire']):
    """
    Create scatter plot for storm data @212042.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=x_lab, y=y_lab, data=storm_data[storm_data.Datetime < date_postfire], \
                    color=colors[0], marker=marker, alpha = alpha, label = labels[0])
    sns.scatterplot(x=x_lab, y=y_lab, data=storm_data[storm_data.Datetime >= date_postfire], \
                    color=colors[1], marker=marker, alpha = alpha, label = labels[1])
    # Add title and labels
    plt.title(f'StormEvent Data of Runoff vs. Turbidity @{site}')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.xscale('log')
    plt.yscale('log')

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'../output/figs/{site}/StormData_Scatter_NTU_Flow@{site}_{freq}_{Q_thr}.png', bbox_inches = 'tight', dpi = 300)