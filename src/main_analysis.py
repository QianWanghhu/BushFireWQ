import numpy as np
import pandas as pd
import os
from functions import EventDataComb, EventFilter

# This is the main script used to process data.
# TODO: Call R package for storm event split.

# Re-filter events based on discharge threshold given.
fn_storm_summary_212042 = '212042_NTU_StormEventSummaryData.csv'
storm_info = pd.read_csv('../output/CQ_analysis/212042/' + fn_storm_summary_212042, index_col = 'Unnamed: 0')
fn_data_212042 = '212042_Hourly.csv'
data_212042 = pd.read_csv('../output/' + fn_data_212042, index_col = 'id')

# Convert data_212042['Datatime'] as datetime
data_212042['Datetime'] = pd.to_datetime(data_212042['Datetime'], format = '%d/%m/%Y %H:%M')
# Add " 00:00:00" to date without hour info for consistency. 
storm_info.reset_index(inplace = True, drop=True)
storm_info['start'] = [date + ' 00:00:00' if ' ' not in date else date for date in storm_info['start']]
storm_info['start'] = pd.to_datetime(storm_info['start'], format = '%Y-%m-%d %H:%M:%S')
storm_info['end'] = [date + ' 00:00:00' if ' ' not in date else date for date in storm_info['end']]
storm_info['end'] = pd.to_datetime(storm_info['end'], format = '%Y-%m-%d %H:%M:%S')

# Re-filter events using data_212042 and storm_info
event_info = EventFilter(data_212042, storm_info, 0.5, 'Discharge (cms)')
storm_df = EventDataComb(data_212042, event_info)
storm_df.index.name = 'id'

# Save files
event_info.to_csv('../output/CQ_analysis/212042/' + fn_storm_summary_212042)
storm_df.to_csv('../output/CQ_analysis/212042/' + '212042_StormEventRefilterData.csv')