import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sci_opt
from scipy.interpolate import interp1d

# This file contains functions for data analysis
def EventFilter(data, event_info, Qthresh, q_name, time_freq):
    """Storm events are further filtered based on the flow peak.
    The original events are identified by R packages."""
    # Loop over events to get peak flow. 
    for ii in range(event_info.shape[0]):
        if time_freq == 'D':
            start_time, end_time = pd.Timestamp(event_info.start[ii].date()), pd.Timestamp(event_info.end[ii].date())
            # Filter data according to start and end time.
            filtered_df = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)]
            if start_time == end_time:
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
        # Get rising and falling limb
    event_info[f'Event_filter_{Qthresh}'] = np.where(event_info['q_peak'] >= Qthresh, 1, 0)
    return event_info

def EventDataComb(data, event_info, baseflow, Qthresh, time_freq):
    """Function for Combine all C and Q data for given events. 
    -- Return Dataframe with Datetime as index; flow as the first column and conc as the second column.
        StormEventID as the third column.
    """
    storm_df = pd.DataFrame(data = None, columns = data.columns)
    cols_new = ['stormID', 'stormCount', 'base_flow', 'storm_flow', 'total_flow']
    storm_df[cols_new] = None
    storm_limbs = {'risingLimbs': {}, 'fallingLimbs': {}}
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
            for col in cols_new[-3:]:
                filtered_df[col] = \
                    baseflow[(baseflow['datetime'] >= start_time) & (baseflow['datetime'] <= end_time)].loc[:, col].values
                    
            storm_df = pd.concat([storm_df, filtered_df], axis=0)
            filtered_df.loc[:, 'norm_tot_q'] = (filtered_df['total_flow'] - filtered_df['total_flow'].min()) / (filtered_df['total_flow'].max() - filtered_df['total_flow'].min())
            filtered_df.loc[:, 'norm_c'] = (filtered_df['Turbidity (NTU)'] - filtered_df['Turbidity (NTU)'].min()) / (filtered_df['Turbidity (NTU)'].max() - filtered_df['Turbidity (NTU)'].min())
            filtered_df.loc[:, 'norm_ts'] = (filtered_df['Datetime'] - filtered_df['Datetime'].min()) / (filtered_df['Datetime'].max() - filtered_df['Datetime'].min())
            # Calculate normalized storm flows, rising and falling limbs, and norm conc and time.
            storm_limbs = processStormEventsWithConc(filtered_df, storm_limbs)
    return storm_df, storm_limbs

def EventSmooth(event_info):
    """
    This function combines two or more event with time lag <= 24 hrs as one event.
    The number of events largely reduces.
    """
    # Sample data (replace this with your actual DataFrame)
    # Initialize a list to store combined events
    combined_events = []
    # Initialize the first event to compare with
    current_event = event_info.iloc[0]
    for i in range(1, len(event_info)):
        next_event = event_info.iloc[i]        
        # Check if the time difference between events is less than 24 hours
        if (next_event['start'] - current_event['end']).total_seconds() / 3600 <= 24:
            # Combine events: take the earliest start and the latest end time
            current_event['end'] = max(current_event['end'], next_event['end'])
            current_event['q_peak'] = max(current_event['q_peak'], next_event['q_peak'])  # Assuming you want the max peakflow
        else:
            # Add the current combined event to the list and update the current event
            combined_events.append(current_event)
            current_event = next_event

    # Don't forget to append the last event
    combined_events.append(current_event)
    # Convert the combined events back to a DataFrame
    combined_df = pd.DataFrame(combined_events)
    kk = 0
    for ii in list(combined_df.index):
        combined_df.loc[ii, 'duration_hrs'] = \
            (combined_df.loc[ii, 'end'] - combined_df.loc[ii, 'start']).total_seconds() / 3600
        combined_df.loc[ii, 'stormID'] = kk
        kk += 1
    # Reset index for the final DataFrame
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df

def processStormEventsWithConc(storm_df, storm_limbs):
    """
    calculate rising, falling, norm_tot_q, norm_c, norm_ts
    """
    stormID = storm_df.loc[:, 'stormID'].unique()[0]
    peak_date = storm_df.loc[storm_df['total_flow'].idxmax(), 'Datetime']
    storm_limbs['risingLimbs'][stormID] = storm_df[storm_df['Datetime'] <= peak_date]
    storm_limbs['fallingLimbs'][stormID] = storm_df[storm_df['Datetime'] > peak_date]
    return storm_limbs

def ProcessHysteresis(storm_df, storm_limbs):
    # Create an empty DataFrame to store hysteresis data
    hysteresis_data = pd.DataFrame(columns=['q_quant', 'risingerp', 'fallingerp', '\
                                            hyst_index', 'flsh_index', 'run_id'])
    # Iterate over storms within each run
    for jj in storm_df.stormID.unique():
        # Sort out rising limb data
        rising_limb_data = storm_limbs['risingLimbs'][jj]
        q_norm_rising = rising_limb_data['norm_tot_q']
        c_norm_rising = rising_limb_data['norm_c']
        
        # Sort out falling limb data
        falling_limb_data = storm_limbs['fallingLimbs'][jj]
        q_norm_falling = falling_limb_data['norm_tot_q']
        c_norm_falling = falling_limb_data['norm_c']  
        # Ensure there are at least 2 points to interpolate per event
        len_rising = len(c_norm_rising)
        len_falling = len(c_norm_falling)
        
        if (len_rising > 1 and len_falling > 1 and 
            len(np.unique(q_norm_rising)) > 1 and 
            len(np.unique(q_norm_falling)) > 1):        
            # Perform interpolation
            interp_rising = interp1d(q_norm_rising, c_norm_rising, kind='linear',\
                                      bounds_error=False, fill_value=np.nan)
            interp_falling = interp1d(q_norm_falling, c_norm_falling, kind='linear', \
                                      bounds_error=False, fill_value=np.nan)
            xForInterp = np.arange(0, 1.01, 0.01)
            # Interpolate at xForInterp
            rising_interp_vals = interp_rising(xForInterp)
            falling_interp_vals = interp_falling(xForInterp)
            
            # Combine data and calculate c-Q indices
            cQ_interp = pd.DataFrame({
                'q_quant': xForInterp,
                'risingerp': rising_interp_vals,
                'fallingerp': falling_interp_vals
            })
            
            # Calculate hysteresis index
            cQ_interp['hyst_index'] = cQ_interp['risingerp'] - cQ_interp['fallingerp']
            
            # Calculate flushing index
            flushing_index = storm_limbs['risingLimbs'][jj]['norm_c'].values[-1] - \
                storm_limbs['risingLimbs'][jj]['norm_c'].values[0]
            cQ_interp['flsh_index'] = flushing_index
            
            # Add run_id and storm_id
            cQ_interp['stormID'] = jj
            
            # Append to the hysteresis data
            hysteresis_data = pd.concat([hysteresis_data, cQ_interp], ignore_index=True)    
        hysteresis_data.index.name = 'id'
    return hysteresis_data

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