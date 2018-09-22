"""
Script is Created by Tiantian Feng
"""

import os, errno
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import matplotlib.pyplot as plt

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

sys.path.append(os.path.join(os.path.curdir, '../', 'util'))
from load_data_basic import *
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn import preprocessing
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA


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
        parser.add_argument('-c', '--cluster', type=str, required=True,
                            help='number of cluster')
        parser.add_argument('-h', '--hour_window', type=str, required=True,
                            help='number of cluster')
        
        args = parser.parse_args()
        
        """
            main_data_directory = '../../data/keck_wave1/2_preprocessed_data'
            window, step = 60, 30
        """
        
        main_data_directory = os.path.join(os.path.expanduser(os.path.normpath(args.main_data_directory)),
                                           'keck_wave2/2_preprocessed_data')
        
        window, step, n_cluster, hour_window = int(args.window), int(args.step), int(args.cluster), int(args.hour_window)
    
    else:
        main_data_directory = '../../data'
        window, step, n_cluster, hour_window = 60, 30, 2, 4
        
    print('----------------------------------------------------------------')
    print('main_data_directory: ' + main_data_directory)
    print('----------------------------------------------------------------')
    
    return main_data_directory, window, step, n_cluster, hour_window


def compute_stat(session_name, data, output_data_df, feat, stats_col, threshold=12):
    if len(data) > threshold:
        for col in stats_col:
            if col == 'mean':
                output_data_df[session_name + '_' + feat + '_' + col] = np.mean(data)
            elif col == 'std':
                output_data_df[session_name + '_' + feat + '_' + col] = np.std(data)
            elif col == 'min':
                output_data_df[session_name + '_' + feat + '_' + col] = np.min(data)
            elif col == 'max':
                output_data_df[session_name + '_' + feat + '_' + col] = np.max(data)
            elif col == 'median':
                output_data_df[session_name + '_' + feat + '_' + col] = np.median(data)
            elif col == 'quantile25':
                output_data_df[session_name + '_' + feat + '_' + col] = np.percentile(data, 25, axis=0)
            elif col == 'quantile75':
                output_data_df[session_name + '_' + feat + '_' + col] = np.percentile(data, 75, axis=0)
            elif col == 'skew':
                output_data_df[session_name + '_' + feat + '_' + col] = skew(data)
            elif col == 'range':
                output_data_df[session_name + '_' + feat + '_' + col] = np.max(data) - np.min(data)
            elif col == 'kurtosis':
                output_data_df[session_name + '_' + feat + '_' + col] = kurtosis(data)
            elif col == 'fit':
                idx = np.isfinite(data)
                output = np.polyfit(np.arange(0, len(data), 1)[idx], data[idx], 5)
                
                for i in range(6):
                    output_data_df[session_name + '_' + feat + str(i) + '_' + col] = output[i]
    
    else:
        for col in stats_col:
            output_data_df[session_name + '_' + feat + '_' + col] = np.nan
    
    return output_data_df


def extract_feat_and_save(frame_om_df, frame_om_preprocess_df,
                          mgt_df, participant_id, shift, survey_date):
    
    # Read basic
    survey_time = pd.to_datetime(mgt_df.index.values[0]).strftime(date_time_format)[:-3]
    return_df = pd.DataFrame(index=[survey_time])

    stats_col = ['mean', 'std', 'max', 'min', 'range']
    prefix_name = 'feat'

    # Copy basic contex
    return_df['participant_id'] = participant_id
    return_df['shift'] = shift
    return_df['survey_time'] = survey_time
    
    # Copy some mgt data
    copy_col = ['cluster', 'stress_mgt', 'anxiety_mgt', 'pos_af_mgt', 'neg_af_mgt']

    for col in copy_col:
        return_df[col] = mgt_df[col].values[0]

    print('Participant: %s, date: %s' % (participant_id, survey_date))
    
    # 1. Compute stats on ready-to-use feature
    physio_col = ['AvgBreathingRate', 'StdDevBreathingRate',
                  'AvgBreathingDepth', 'StdDevBreathingDepth',
                  'AvgGForce', 'StdDevGForce']
    
    for col in physio_col:
        data_array = frame_om_df[col].dropna()
        data_array = data_array[data_array != 0]
        return_df = compute_stat(prefix_name, np.array(data_array),
                                 return_df, col, stats_col, threshold=12)
    
    # 2. Steps
    return_df = compute_stat(prefix_name, np.array(frame_om_preprocess_df['Steps'].dropna()),
                             return_df, 'Steps', stats_col)
    
    # 3. Heart Rate
    array_heart_rate = frame_om_preprocess_df['HeartRate_mean'].dropna()
    cond1 = array_heart_rate > 40
    cond2 = array_heart_rate < 150
    return_df = compute_stat(prefix_name, np.array(array_heart_rate[cond1 & cond2].dropna()),
                             return_df, 'HeartRate', stats_col)
    
    # 4. HRV feature, choose rr peak coverage region above 0.8
    om_hrv_df = frame_om_df[frame_om_df['RRPeakCoverage'] > 0.8]
    om_hrv_rmsdd = np.sort(om_hrv_df['RMSStdDev_ms'].dropna())
    om_hrv_rrstd = np.sort(om_hrv_df['SDNN_ms'].dropna())
    
    return_df = compute_stat(prefix_name, om_hrv_rmsdd, return_df, 'RMSStdDev_ms', stats_col)
    return_df = compute_stat(prefix_name, om_hrv_rrstd, return_df, 'SDNN_ms', stats_col)
    
    return return_df


def extract_feat_with_survey(UserInfo, MGT_df, hour_window=4, window=60, step=30):
    
    output_path = '../output/ml_feat'
    
    user_index = 0
    final_df = pd.DataFrame()

    # Preprocessed feature path for participant
    window_path = os.path.join('../output/preprocessed_data', 'window_' + str(window) + '_step_' + str(step))
    if os.path.exists(window_path) is False:
        os.mkdir(window_path)
    
    # Iterate rows
    for uid, data in UserInfo.iterrows():
        print('----------------------------------------------------------------')
        print('Participant completed: %d' % (user_index))
        user_index = user_index + 1
        print('----------------------------------------------------------------')
        
        # Get participant id and shift type
        participant_id, shift_type = data['ParticipantID'], data['Shift']

        # Read MGT data
        cond1 = MGT_df['uid'] == uid
        participantMGT = MGT_df.loc[cond1]
        
        # Read the OM signal data
        om_file_path = os.path.join(main_data_directory, 'keck_wave2/3_preprocessed_data', 'omsignal',
                                    participant_id + '_omsignal.csv')
        om_preprocess_file_path = os.path.join(window_path, participant_id + '.csv')
        
        # If om file exist
        if os.path.exists(om_file_path) is True and os.path.exists(om_preprocess_file_path) is True:
            
            # Read om data and om preprocessed data
            om_df = pd.read_csv(om_file_path, index_col=0)
            om_df = om_df.sort_index()
            
            om_preprocess_df = pd.read_csv(om_preprocess_file_path, index_col=0)
            om_preprocess_df = om_preprocess_df.sort_index()
            
            # Iterate MGT per participant
            if len(participantMGT) > 0:
                for timestamp, dailyMGT in participantMGT.iterrows():
                
                    # Define parameters
                    survey_date = pd.to_datetime(timestamp).strftime(date_only_date_time_format)
                    threshold = int(hour_window / 2) * 3600
                
                    # Get daily MGT
                    frame_MGT = dailyMGT.to_frame().transpose()
                    survey_time = pd.to_datetime(frame_MGT.index.values[0]).strftime(date_time_format)
                    
                    # om start time is survey time minus hours we input here, end time is just survey data
                    frame_start = (pd.to_datetime(survey_time) - timedelta(hours=hour_window)).strftime(date_time_format)[:-3]
                    frame_stop = (pd.to_datetime(survey_time)).strftime(date_time_format)[:-3]
                    
                    # Get the om data in the defined time frame
                    frame_om_raw_data_df = om_df[frame_start:frame_stop]
                    frame_om_preprocess_df = om_preprocess_df[frame_start:frame_stop]
                
                    # At least 50 % of data, then process
                    if len(frame_om_raw_data_df) > threshold:
                        feature_and_survey = extract_feat_and_save(frame_om_raw_data_df, frame_om_preprocess_df,
                                                                   dailyMGT.to_frame().transpose(),
                                                                   participant_id, shift_type, survey_date)
                        
                        # ADD YOUR CODE HERE IF YOU WANT TO EXTRACT MORE FEATURES
                        # JUST APPEND TO feature_and_survey, BEFORE feature_and_survey HAS BEEN APPEND TO final_df
                        # ------------------------------------------

                        # ------------------------------------------
    
                        final_df = final_df.append(feature_and_survey)
                        final_df.to_csv(os.path.join(output_path, 'ml_input_feat.csv'))
                    
        print('MGT number: %d;' % (len(participantMGT)))
        print('----------------------------------------------------------------')


def select_scaler(scaler_name):
    # Transformation
    if scaler_name == 'z_norm':
        scaler = preprocessing.StandardScaler()
    elif scaler_name == 'min_max':
        scaler = preprocessing.MinMaxScaler()
    else:
        scaler = preprocessing.Normalizer()
    
    return scaler


def append_cluster_MGT(UserInfo, MGT_df, n_cluster=2):
    
    # First append MGT
    final_MGT_df = pd.DataFrame()
    for index, data in UserInfo.iterrows():
        user_MGT_col = MGT_df.loc[MGT_df['uid'] == index]
        final_MGT_df = final_MGT_df.append(user_MGT_col)
    
    # 1. Take only affect labels at work
    col = ['stress_mgt', 'anxiety_mgt', 'neg_af_mgt', 'pos_af_mgt']
    
    final_MGT_df = final_MGT_df.loc[(final_MGT_df['location_mgt'] == 2) | (final_MGT_df['itp_mgt'] > -1)]
    final_MGT_df = final_MGT_df.dropna(subset=col)
    
    # 2. Normalization
    scaler_name = 'norm'
    affect_label_scaler = select_scaler(scaler_name)
    norm_affect_label_array = affect_label_scaler.fit_transform(np.array(final_MGT_df[col]))
    
    # 3. PCA on normalization 4-d vectors
    pca = PCA(n_components=2)
    pca_result = pca.fit(norm_affect_label_array).transform(norm_affect_label_array)
    
    # 4. Normalization 2-d PCA vectors
    scaler_name = 'z_norm'
    pca_scaler = select_scaler(scaler_name)
    norm_pca_components = pca_scaler.fit_transform(pca_result)
    
    # 5. Cluster using K-Means
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, tol=1e-4, max_iter=1000).fit(norm_pca_components)
    cluster = np.array(kmeans.predict(norm_pca_components))
    final_MGT_df['cluster'] = cluster
    
    return final_MGT_df


if __name__ == "__main__":
    
    if os.path.exists(os.path.join('../output')) is False:
        os.mkdir(os.path.join('../output'))
    
    if os.path.exists(os.path.join('../output/ml_feat')) is False:
        os.mkdir(os.path.join('../output/ml_feat'))

    # Read args
    main_data_directory, window, step, n_cluster, hour_window = parseArgs()
    
    # Read MGT and user level information
    UserInfo = read_AllBasic(main_data_directory)
    MGT_df = read_MGT(main_data_directory)
    
    print('----------------------------------------------------------------')
    print('Number of user in total: %d' % (len(UserInfo)))
    print('----------------------------------------------------------------')
    
    participant_timeline = pd.DataFrame()
    UserInfo = UserInfo[:]
    
    # Append clustering
    final_MGT_df = append_cluster_MGT(UserInfo, MGT_df, n_cluster=2)

    # Extract Feature
    extract_feat_with_survey(UserInfo, final_MGT_df)