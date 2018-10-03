import os, errno
import pandas as pd
import pdb
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
        job_shift = job_shift_df.loc[job_shift_df['ID'] == id_data['user_id']]['job_shift'].values[0]
        participant_id = id_data['user_id']
        
        frame_df = pd.DataFrame(job_shift, index=['job_shift'], columns=[participant_id]).transpose()
        
        participant_id_job_shift_df = frame_df if len(
            participant_id_job_shift_df) == 0 else participant_id_job_shift_df.append(frame_df)
    
    return participant_id_job_shift_df


def read_participant_info(main_data_directory, index=1):
    
    IDs = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/id-mapping', 'mitreids.csv'))
    participant_info = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/participant_info', 'participant_info.csv'))
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
    affect_df = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/ground_truth/MGT', 'pan.d.csv'))
    affect_df.index = pd.to_datetime(affect_df.index)
    affect_df = affect_df[['ID', 'start', 'pos.affect.d', 'neg.affect.d']]

    anxiety_df = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/ground_truth/MGT', 'anxiety.d.csv'))
    anxiety_df.index = pd.to_datetime(anxiety_df.index)
    anxiety_df = anxiety_df[['ID', 'start', 'anxiety.d']]

    stress_df = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/ground_truth/MGT', 'stress.d.csv'))
    stress_df.index = pd.to_datetime(stress_df.index)
    stress_df = stress_df[['ID', 'start', 'stress.d']]

    itp_df = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/ground_truth/MGT', 'itp.d.csv'))
    itp_df.index = pd.to_datetime(itp_df.index)
    itp_df = itp_df[['ID', 'start', 'work_status']]

    context_df = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/ground_truth/MGT', 'merged.daily.no_free_response.csv'))
    context_df.index = pd.to_datetime(context_df.index)
    context_df = context_df[['Name', 'StartDate', 'context3']]

    #work_df = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/ground_truth/MGT', 'worktoday.csv'), index_col=3)
    #work_df.index = pd.to_datetime(work_df.index)
    #work_df = work_df[['ID', 'end', 'work_status']]

    MGT = pd.merge(affect_df, anxiety_df, left_on=['ID', 'start'], right_on=['ID', 'start'], how='outer')
    MGT = pd.merge(MGT, stress_df, left_on=['ID', 'start'], right_on=['ID', 'start'], how='outer')
    MGT = pd.merge(MGT, itp_df, left_on=['ID', 'start'], right_on=['ID', 'start'], how='outer')
    MGT = pd.merge(MGT, context_df, left_on=['ID', 'start'], right_on=['Name', 'StartDate'], how='outer')
    MGT = MGT.set_index('start')

    MGT = MGT.drop(['Name', 'StartDate'], axis=1)
    
    return MGT


# Load pre study data
def read_pre_study_info(main_data_directory):
    PreStudyInfo = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/participant_info', 'prestudy_data.csv'), index_col=3)
    PreStudyInfo.index = pd.to_datetime(PreStudyInfo.index)
    
    return PreStudyInfo


# Load IGTB data
def read_IGTB(main_data_directory):
    IGTB = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/ground_truth/IGTB', 'igtb_composites.csv'), index_col=1)
    IGTB.index = pd.to_datetime(IGTB.index)
    
    return IGTB


# Load IGTB data
def read_Demographic(main_data_directory):
    DemoGraphic = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/demographic', 'Demographic.csv'))
    DemoGraphic.index = pd.to_datetime(DemoGraphic.index)
    
    return DemoGraphic


# Load IGTB data
def read_IGTB_Raw(main_data_directory):
    IGTB = pd.read_csv(os.path.join(main_data_directory, 'keck_wave3/ground_truth', 'IGTB_R.csv'), index_col=False)
    IGTB.index = pd.to_datetime(IGTB.index)
    
    return IGTB


# Load all per user level information
def read_user_information(main_data_directory):
    
    # Read participant information
    participant_info = read_participant_info(main_data_directory)
    
    # Read Pre-Study info
    PreStudyInfo = read_pre_study_info(main_data_directory)

    # Read IGTB info
    IGTB = read_IGTB(main_data_directory)

    # Demographic
    # Demographic = read_Demographic(main_data_directory)

    # Merge different df together
    UserInfo = pd.merge(IGTB, PreStudyInfo, left_on='ID', right_on='redcap_survey_identifier', how='outer')
    UserInfo = pd.merge(UserInfo, participant_info, left_on='ID', right_on='MitreID', how='outer')
    # UserInfo = pd.merge(UserInfo, Demographic, left_on='uid', right_on='uid', how='outer')
    
    # Choose waves
    # Example, read only wave 1 and wave 2
    # UserInfo = UserInfo.loc[UserInfo['Wave'] != 3]

    UserInfo = UserInfo.set_index('ID')

    # 1, Registered Nurse(RN)
    # 2, Certified Nursing Assistant(CNA)
    # 3, Monitor Tech(MT)
    # 4, Physical Therapist
    # 5, Occupational Therapist
    # 6, Speech Therapist
    # 7, Respiratory Therapist
    # 8, Other
    #
    # Example, read only nurses:
    # UserInfo = UserInfo.loc[(UserInfo['current_position_pre-study'] == 1) |
    #                         (UserInfo['current_position_pre-study'] == 2)]
    
    return UserInfo

