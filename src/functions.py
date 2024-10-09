import numpy as np
import pandas as pd
import os
# This file contains functions for data analysis
def EventFilter(data, event_info, Qthresh, q_name):
    """Storm events are further filtered based on the flow peak. The original events are identified by R packages."""
    # Loop over events to get peak flow. 
    for ii in range(event_info.shape[0]):
        start_time, end_time = event_info.start[ii], event_info.end[ii]
        # Filter data according to start and end time.
        filtered_df = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)]
        event_info.loc[ii, 'q_peak'] = filtered_df[q_name].max()
    event_info['Event_filter'] = np.where(event_info['q_peak'] >= Qthresh, 1, 0)
    return event_info  

def EventDataComb(data, event_info):
    """Function for Combine all C and Q data for given events. 
    -- Return Dataframe with Datetime as index; flow as the first column and conc as the second column.
        StormEventID as the third column.
    """
    storm_df = pd.DataFrame(data = None, columns = data.columns)
    for ii in range(event_info.shape[0]):
        if event_info.loc[ii, 'Event_filter']:
            start_time, end_time = event_info.start[ii], event_info.end[ii]
            filtered_df = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)]
            storm_df = pd.concat([storm_df, filtered_df])
    return storm_df

def CQModel():
    "Function to build CQ model"
    pass