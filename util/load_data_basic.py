import os, errno
import pandas as pd
from datetime import datetime, timedelta

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'


def getParticipantIDJobShift(main_data_directory):
    participant_id_job_shift_df = []
    
    # job shift
    job_shift_df = pd.read_csv(os.path.join(main_data_directory, 'job shift/Job_Shift.csv'))
    
    # read id
    id_data_df = pd.read_csv(os.path.join(main_data_directory, 'ground_truth/IDs.csv'))
    
    for index, id_data in id_data_df.iterrows():
        # get job shift and participant id
        job_shift = job_shift_df.loc[job_shift_df['uid'] == id_data['user_id']]['job_shift'].values[0]
        participant_id = id_data['user_id']
        
        frame_df = pd.DataFrame(job_shift, index=['job_shift'], columns=[participant_id]).transpose()
        
        participant_id_job_shift_df = frame_df if len(
            participant_id_job_shift_df) == 0 else participant_id_job_shift_df.append(frame_df)
    
    return participant_id_job_shift_df


def getParticipantInfo(main_data_directory, index=1):
    
    IDs = pd.read_csv(os.path.join(main_data_directory, 'keck_wave2/id-mapping', 'mitreids.csv'))
    participant_info = pd.read_csv(os.path.join(main_data_directory, 'keck_wave2/participant_info', 'participant_info.csv'))
    participant_info = participant_info.fillna("")
    
    for index, row in participant_info.iterrows():
        
        participant_id = row['ParticipantID']
        mitre_id = row['MitreID']
        
        participant_info.loc[index, 'MitreID'] = IDs.loc[IDs['participant_id'] == participant_id]['mitre_id'].values[0]

    # IDs.index.names = ['MitreID']
    # participant_info = participant_info.set_index('MitreID')
    
    return participant_info

# start date, end date for wave 1 and pilot
def getParticipantStartTime():
    return datetime(year=2018, month=2, day=20)


def getParticipantEndTime():
    return datetime(year=2018, month=6, day=10)


# Load mgt data
def read_MGT(main_data_directory):
    MGT = pd.read_csv(os.path.join(main_data_directory, 'keck_wave2/ground_truth', 'MGT.csv'), index_col=2)
    MGT.index = pd.to_datetime(MGT.index)
    
    return MGT


# Load pre study data
def read_pre_study_info(main_data_directory):
    PreStudyInfo = pd.read_csv(os.path.join(main_data_directory, 'keck_wave2/ground_truth', 'Pre-Study.csv'), index_col=1)
    PreStudyInfo.index = pd.to_datetime(PreStudyInfo.index)
    
    return PreStudyInfo


# Load IGTB data
def read_IGTB(main_data_directory):
    IGTB = pd.read_csv(os.path.join(main_data_directory, 'keck_wave2/ground_truth', 'IGTB.csv'), index_col=1)
    IGTB.index = pd.to_datetime(IGTB.index)
    
    return IGTB


# Load IGTB data
def read_Demographic(main_data_directory):
    DemoGraphic = pd.read_csv(os.path.join(main_data_directory, 'keck_wave2/demographic', 'Demographic.csv'))
    DemoGraphic.index = pd.to_datetime(DemoGraphic.index)
    
    return DemoGraphic


# Load IGTB data
def read_IGTB_Raw(main_data_directory):
    IGTB = pd.read_csv(os.path.join(main_data_directory, 'keck_wave2/ground_truth', 'IGTB_R.csv'), index_col=False)
    IGTB.index = pd.to_datetime(IGTB.index)
    
    return IGTB


# Load all basic data
def read_AllBasic(main_data_directory):
    
    # Read participant information
    participant_info = getParticipantInfo(main_data_directory)
    
    # Read MGT
    MGT = read_MGT(main_data_directory)

    # Read Pre-Study info
    PreStudyInfo = read_pre_study_info(main_data_directory)

    # Read IGTB info
    IGTB = read_IGTB(main_data_directory)

    # Demographic
    Demographic = read_Demographic(main_data_directory)

    UserInfo = pd.merge(IGTB, PreStudyInfo, left_on='uid', right_on='uid', how='outer')
    UserInfo = pd.merge(UserInfo, participant_info, left_on='uid', right_on='MitreID', how='outer')
    UserInfo = pd.merge(UserInfo, Demographic, left_on='uid', right_on='uid', how='outer')
    # UserInfo = UserInfo.loc[UserInfo['Wave'] != 3]

    UserInfo = UserInfo.set_index('uid')

    UserInfo = UserInfo.loc[(UserInfo['current_position_pre-study'] == 1) |
                            (UserInfo['current_position_pre-study'] == 2)]
    
    return UserInfo

