import numpy as np
import pandas as pd
import os
from functions import EventDataComb, EventFilter, EventSmooth, ProcessHysteresis, cal_cv

# This is the main script used to process data.
# TODO: Call R package for storm event split.
data_freq = 'H'
site = 212042
# Re-filter events based on discharge threshold given.
fn_storm_summary_212042 = f'{site}_NTU_StormEventSummaryData.csv'
dir_output = f'../output/CQ_analysis/{site}/'
storm_info = pd.read_csv(dir_output + fn_storm_summary_212042, index_col = 'id')
if data_freq == 'H':
    fn_data_212042 = f'{site}_Hourly.csv'
    data_212042 = pd.read_csv('../output/' + fn_data_212042, index_col = 'id').dropna()
    # Convert data_212042['Datatime'] as datetime
    data_212042['Datetime'] = pd.to_datetime(data_212042['Datetime'], format = '%d/%m/%Y %H:%M')
    # Add " 00:00:00" to date without hour info for consistency. 
    storm_info.reset_index(inplace = True, drop=True)
    storm_info['start'] = [date + ' 00:00' if ' ' not in date else date for date in storm_info['start']]
    storm_info['start'] = pd.to_datetime(storm_info['start'], format = 'mixed', dayfirst=True)
    storm_info['end'] = [date + ' 00:00' if ' ' not in date else date for date in storm_info['end']]
    storm_info['end'] = pd.to_datetime(storm_info['end'], format = 'mixed', dayfirst=True)
    storm_info.index.name = 'id'
    # Re-filter events using data_212042 and storm_info
    Q_thred_filter = [0.6]
    for Q_thr in Q_thred_filter:
        event_info = EventFilter(data_212042, storm_info, Q_thr, 'Discharge (cms)', data_freq)
        event_info.to_csv(dir_output + fn_storm_summary_212042)
        # Combine events of time lag > 24 hours
        event_comb = EventSmooth(event_info[event_info[f'Event_filter_{Q_thr}'] == 1]) # Dataframe with peak > 2 m3/s
        event_comb.index.name = 'id'
        event_comb.to_csv(dir_output + 'QAbove_' + str(Q_thr) + f'_{site}_StormEventClean.csv')
    # Read dataframe containing baseflow
    baseflow = pd.read_csv(f'../output/CQ_analysis/{site}/' + f'{site}_NTU_DischargeData.csv', index_col = 'Unnamed: 0')
    baseflow['datetime'] = pd.to_datetime(baseflow['datetime'], format = 'mixed', dayfirst=True)
    # Save files
    for Q_thr in Q_thred_filter:
        storm_df, storm_limbs = EventDataComb(data_212042, event_comb, baseflow, Q_thr, data_freq)
        storm_df.index.name = 'id'
        storm_df.to_csv(dir_output + 'Q_above_' + str(Q_thr) + f'_{site}_StormEventRefilterData.csv')
        cv_df = cal_cv(storm_df)
        cv_df.to_csv(f'{dir_output}/CV_flow_tbdt.csv')
        hysteresis_data = ProcessHysteresis(event_comb, storm_limbs)
    hysteresis_data.to_csv(f'../output/CQ_analysis/{site}/' + 'HysteresisEventClean.csv')
elif data_freq == 'D':
    fn_data_212042 = f'{site}_Daily.csv'
    data_212042 = pd.read_csv('../output/' + fn_data_212042, index_col = 'id')

    # Convert data_212042['Datatime'] as datetime
    data_212042['Datetime'] = pd.to_datetime(data_212042['Datetime'], format = '%d/%m/%Y')
    # Add " 00:00:00" to date without hour info for consistency. 
    storm_info.reset_index(inplace = True, drop=True)
    storm_info['start'] = [date + ' 00:00' if ' ' not in date else date for date in storm_info['start']]
    storm_info['start'] = pd.to_datetime(storm_info['start'], format = 'mixed', dayfirst=True)
    storm_info['end'] = [date + ' 00:00' if ' ' not in date else date for date in storm_info['end']]
    storm_info['end'] = pd.to_datetime(storm_info['end'], format = 'mixed', dayfirst=True)
    storm_info.index.name = 'id'

    # Re-filter events using data_212058 and storm_info
    Q_thred_filter = [1]
    for Q_thr in Q_thred_filter:
        event_info = EventFilter(data_212042, storm_info, Q_thr, 'Discharge (cms)', data_freq)
    # Save event summary using daily data to process.
    event_info.to_csv(dir_output + 'Daily' + fn_storm_summary_212042)

    # Combine events of time lag > 24 hours
    event_comb = EventSmooth(event_info[event_info['Event_filter_2'] == 1]) # Dataframe with peak > 2 m3/s
    event_comb.to_csv(dir_output + 'QAbove_' + str(Q_thr) + f'_{site}_StormEventClean.csv')

    # Save files
    for Q_thr in Q_thred_filter:
        storm_df = EventDataComb(data_212042, event_info, Q_thr, data_freq)
        storm_df.index.name = 'id'
        storm_df.to_csv(dir_output + 'DailyQ_above_' + str(Q_thr) + f'_{site}_StormEventRefilterData.csv')
else:
    print('The frequency is not supported in this analysis.')