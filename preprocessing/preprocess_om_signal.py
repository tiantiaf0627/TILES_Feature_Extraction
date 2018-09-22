"""
Script is created by Tiantian Feng
"""

import os, errno
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

sys.path.append(os.path.join(os.path.curdir, '../', 'util'))
from load_data_basic import *


def parseArgs():
    DEBUG = 1
    
    if DEBUG == 0:
        """
            Parse the args:
            1. main_data_directory: directory to store keck data
            2. output_directory: main output directory

        """
        parser = argparse.ArgumentParser(description='Create a dataframe of worked days.')
        parser.add_argument('-i', '--main_data_directory', type=str, required=True,
                            help='Directory for data.')
        parser.add_argument('-o', '--output_directory', type=str, required=True,
                            help='Directory for output.')
        parser.add_argument('-w', '--window', type=str, required=True,
                            help='moving window')
        parser.add_argument('-s', '--step', type=str, required=True,
                            help='moving step')
        
        args = parser.parse_args()
        
        """
            main_data_directory = '../../data/keck_wave3/3_preprocessed_data'
            window, step = 60, 30
        """
        
        main_data_directory = os.path.join(os.path.expanduser(os.path.normpath(args.main_data_directory)),
                                           'keck_wave3/3_preprocessed_data')

        window, step = int(args.window), int(args.step)
        
    else:
        main_data_directory = '../../data'
        window, step = 60, 30
        
    print('----------------------------------------------------------------')
    print('main_data_directory: ' + main_data_directory)
    print('----------------------------------------------------------------')
    
    return main_data_directory, window, step


def process_and_save_data(data_df, window_output_path, participant_id, window=30, step=15):
    
    # Remove some artifacts in the data showing date before 2018
    start_collection_date = datetime(year=2018, month=2, day=15).strftime(date_time_format)[:-3]

    data_df = data_df[start_collection_date:]
    
    # Get start and stop recording time for the whole session
    start_time_data = pd.to_datetime(data_df.index.values[0])
    start_time_data = start_time_data.replace(second=0, microsecond=0)
    
    stop_time_data = pd.to_datetime(data_df.index.values[-1])
    stop_time_data = stop_time_data.replace(second=0, microsecond=0)
    
    # Calculate delta in between
    delta = int((stop_time_data - start_time_data).total_seconds() / step)

    data_preprocess_df = pd.DataFrame()
    
    for i in range(delta):
        # Frame timestamp start and stop
        frame_start = (start_time_data + timedelta(seconds=step * i - window / 2)).strftime(date_time_format)[:-3]
        frame_stop = (start_time_data + timedelta(seconds=step * i + window / 2)).strftime(date_time_format)[:-3]
        frame_current = (start_time_data + timedelta(seconds=step * i)).strftime(date_time_format)[:-3]

        frame_data_df = data_df[frame_start:frame_stop]

        print('Participant: %s, time: %s, complete rate: %.2f' % (window_output_path,
                                                                  frame_current, float(i / delta) * 100))
                
        # if we have at least 50% of data in the frame
        if len(frame_data_df) > window / 2:
            
            # Preprocess frame
            frame_data_preprocess_df = pd.DataFrame(index=[frame_current])
    
            # 1. Breathing
            # array_breathing_rate = frame_data_df['BreathingRate'].dropna()
            # array_breathing_rate = array_breathing_rate[array_breathing_rate != 0]
            
            # array_breathing_depth = frame_data_df['BreathingDepth'].dropna()
            # array_breathing_depth = array_breathing_depth[array_breathing_depth != 0]

            # frame_data_preprocess_df['BreathingRate_mean'] = np.mean(array_breathing_rate)
            # frame_data_preprocess_df['BreathingDepth_mean'] = np.mean(array_breathing_depth)
            
            # 2. Cadence
            # frame_data_preprocess_df['Cadence_mean'] = np.mean(frame_data_df['Cadence'].dropna())
            
            # 3. Intensity
            # frame_data_preprocess_df['Intensity_mean'] = np.mean(frame_data_df['Intensity'].dropna())
            
            # 4. Steps: aggregate
            frame_data_preprocess_df['Steps'] = np.sum(np.array(frame_data_df['Steps'].dropna()))
            
            # 5. Heart Rate
            array_heart_rate = frame_data_df['HeartRate']
            cond1 = array_heart_rate > 40
            cond2 = array_heart_rate < 150
            frame_data_preprocess_df['HeartRate_mean'] = np.mean(array_heart_rate[cond1 & cond2].dropna())

            # 6. Append data
            data_preprocess_df = data_preprocess_df.append(frame_data_preprocess_df)

    # Save the preprocessed data
    data_preprocess_df.to_csv(os.path.join(window_output_path, participant_id + '.csv'))


def preprocess_om(UserInfo, window=60, step=30):
    
    output_path = '../output/preprocessed_data'
    
    # Output path for participant
    window_path = os.path.join(output_path, 'window_' + str(window) + '_step_' + str(step))
    if os.path.exists(window_path) is False:
        os.mkdir(window_path)
    
    user_index = 0
    
    for uid, data in UserInfo.iterrows():
        print('----------------------------------------------------------------')
        print('Participant completed: %d' % (user_index))
        user_index = user_index + 1
        print('----------------------------------------------------------------')
        
        # Get participant id and shift type
        participant_id, shift_type = data['ParticipantID'], data['Shift']
        
        # Read the OM signal data
        om_file_path = os.path.join(main_data_directory, 'keck_wave3/3_preprocessed_data', 'omsignal',
                                    participant_id + '_omsignal.csv')
        
        # If om csv exist, pre_process the data
        if os.path.exists(om_file_path) is True:
            om_df = pd.read_csv(om_file_path, index_col=0)
            om_df = om_df.sort_index()

            process_and_save_data(om_df, window_path, participant_id, window=window, step=step)
        
        print('----------------------------------------------------------------')


if __name__ == "__main__":
    
    if os.path.exists(os.path.join('../output')) is False:
        os.mkdir(os.path.join('../output'))
    
    if os.path.exists(os.path.join('../output/preprocessed_data')) is False:
        os.mkdir(os.path.join('../output/preprocessed_data'))
    
    main_data_directory, window, step = parseArgs()
    
    # Only nurses are selected here
    # Please modify read_AllBasic on your own flavor to get different type of jobs
    UserInfo = read_user_information(main_data_directory)
    MGT_df = read_MGT(main_data_directory)
    
    print('----------------------------------------------------------------')
    print('Number of user in total: %d' % (len(UserInfo)))
    print('----------------------------------------------------------------')
    
    participant_timeline = pd.DataFrame()
    UserInfo = UserInfo[:50]
    
    preprocess_om(UserInfo)