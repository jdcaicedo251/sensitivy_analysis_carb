#!/usr/bin/env python
# coding: utf-8

# # Re-estimate KPIs 
# 
# There are some improvements and changes that are done to the KPIs, and I need to re-estimate some of them. However, all the data to re-estimate them is in the AWS S3 Buckets, and it is very expensive to download all. The idea of this code is to automate the process to estimate and just let it run. 
# 
# For each policy:
# 
# - Download trips, households, persons, and skims tables
# - Run kpy.py
# - Convert result in yaml file
# - Upload modified results to S3 Bucket

# In[1]:


import os
import re
import sys
import yaml
import time
import boto3
import logging
import numpy as np
import pandas as pd
import openmatrix as omx

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# In[2]:


s3 = boto3.client("s3")
def read_csv(fpath, index_col = None, dytpe = None):
    bucket_name = "carb-results"
    obj = s3.get_object(Bucket = bucket_name, Key = fpath)
    
    return pd.read_csv(obj['Body'], index_col = index_col, dtype = dytpe)

def download_s3(local_file_name,s3_bucket,s3_object_key):
    """
    reference:
    https://stackoverflow.com/questions/41827963/
    track-download-progress-of-s3-file-using-boto3-and-callbacks
    """

    meta_data = s3.head_object(Bucket=s3_bucket, Key=s3_object_key)
    total_length = int(meta_data.get('ContentLength', 0))
    downloaded = 0

    def progress(chunk):
        nonlocal downloaded
        downloaded += chunk
        done = int(50 * downloaded / total_length)
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )
        sys.stdout.flush()

    logger.info(f'Downloading {s3_object_key}')
    with open(local_file_name, 'wb') as f:
        s3.download_fileobj(s3_bucket, s3_object_key, f, Callback=progress)
        
def upload_file_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False

    reference: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


# In[15]:


###############
### UTILS ####
##############

def read_policy_settings():
    """ Read policy settings"""
    a_yaml_file = open('policy_settings.yaml')
    settings = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    return settings


def read_yaml(path):
    a_yaml_file = open(path)
    settings = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    return settings


def save_yaml(path, file):
    with open(path, 'w') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)


def get_metric(metric, results):
    values = []
    names = []
    for arg in results:
        values.append(arg[metric])
        names.append(arg['name'])
    return values, names

def build_df_multi(values, names):
    dfs = []
    for v,n in zip(values,names):
        df = pd.DataFrame.from_dict(data = v, orient = 'index')
        df['name'] =  n
        df.reset_index(inplace = True)
        df.columns = ['category','values','name']
        dfs.append(df)
    return pd.concat(dfs)


############################
### PROCESSING RESULTS ####
###########################

def od_matrix_lookup(origin, destination, matrix):
    ''' Returns the distance between origin and estiantion in miles
    Parameters:
    -----------
    - origing: 1-d array-like. origins ID
    - destination: 1- d array_like. destination ID
    - matrix: 2-d array-like. Origin-destiantion matrix for a given metric.
                              Rows are origins, columns are destinations

    Returns:
    1-d array of the origin-destination metric.
    '''
    assert origin.ndim == 1, 'origin should be a 1-d array'
    assert destination.ndim == 1, 'destination should be 1-d array'
    assert matrix.ndim == 2, 'distance matrix should be 2-d array'
    assert origin.shape == destination.shape, 'origin and destination should have the same shape'

    #Transform array-like to numpy array in case they are not
    #Substract 1 because distance matrix starts in ZERO
    origin = np.array(origin) - 1
    destination = np.array(destination) - 1
    return matrix[origin, destination]


def od_matrix__time_lookup(period, mode, origin, destination, matrix):
    ''' Returns the an 0-D pair value by period and mode.
    Parameters:
    -----------
    - perdiod: int.
        - 'EA'= 0
        - 'AM'= 1
        - 'MD'= 2
        - 'PM'= 3
        - 'EV'= 4
    - mode: int.
        -'DRIVEALONEFREE': 0,
        -'DRIVEALONEPAY':1,
        -'SHARED2FREE': 2,
        -'SHARED3FREE': 3,
        -'SHARED2PAY':4,
        -'SHARED3PAY':5,
        -'WALK': 6,
        -'BIKE': 7,
        -'WALK_HVY': 8,
        -'WALK_LOC': 9,
        -'WALK_EXP': 10,
        -'WALK_COM': 11,
        -'WALK_LRF': 12,
        -'DRIVE_HVY': 13,
        -'DRIVE_LOC': 14,
        -'DRIVE_EXP': 15,
        -'DRIVE_COM': 16,
        -'DRIVE_LRF': 17,
        -'TNC_SINGLE': 18,
        -'TNC_SHARED': 19,
        -'TAXI': 20
    - origing: 1-d array-like. origins ID
    - destination: 1- d array_like. destination ID
    - matrix: 4-d array-like. Travel Time skims. Each dimension correspond to:
        - period
        - mode_index
        - origin
        - destiantion

    Returns:
    1-d array of the origin-destination metric.
    '''
    assert origin.ndim == 1, 'origin should be a 1-d array'
    assert destination.ndim == 1, 'destination should be 1-d array'
    assert period.ndim == 1, 'origin should be a 1-d array'
    assert period.ndim == 1, 'destination should be 1-d array'
    assert matrix.ndim == 4, 'distance matrix should be 4-d array'
    assert origin.shape == destination.shape, 'origin and destination should have the same shape'

    #Transform array-like to numpy array in case they are not
    #Substract 1 because distance matrix starts in ZERO
    origin = np.array(origin) - 1
    destination = np.array(destination) - 1
    return matrix[period, mode, origin, destination]

def time_skims(skims):
    """
    Return time skims for each mode of transportation.
    Time Period Index:
    - 'EA'= 0
    - 'AM'= 1
    - 'MD'= 2
    - 'PM'= 3
    - 'EV'= 4
    Mode Index:
    -'DRIVEALONEFREE': 0,
    -'DRIVEALONEPAY':1,
    -'SHARED2FREE': 2,
    -'SHARED3FREE': 3,
    -'SHARED2PAY':4,
    -'SHARED3PAY':5,
    -'WALK': 6,
    -'BIKE': 7,
    -'WALK_HVY': 8,
    -'WALK_LOC': 9,
    -'WALK_EXP': 10,
    -'WALK_COM': 11,
    -'WALK_LRF': 12,
    -'DRIVE_HVY': 13,
    -'DRIVE_LOC': 14,
    -'DRIVE_EXP': 15,
    -'DRIVE_COM': 16,
    -'DRIVE_LRF': 17,
    -'TNC_SINGLE': 0,
    -'TNC_SHARED': 0,
    -'TAXI': 0

    Return:
    - four-dimensional matrix with travel times. (time_period, mode, origin, destination)
    """
    periods = ['EA', 'AM', 'MD', 'PM', 'EV']
    driving_modes = ['SOV', 'SOVTOLL','HOV2','HOV2TOLL', 'HOV3','HOV3TOLL']
    transit_modes = ['HVY','LOC','EXP','COM','LRF']

    time_skims = []
    for period in periods:
        driving_skims = []
        walk_transit = []
        drive_transit = []
        for mode in driving_modes:
            time_mtx = np.array(skims['{0}_TIME__{1}'.format(mode, period)])
            driving_skims.append(time_mtx)

        for mode in transit_modes:
            walk_time_skim = (np.array(skims['WLK_{0}_WLK_WAIT__{1}'.format(mode, period)]) +             np.array(skims['WLK_{0}_WLK_IWAIT__{1}'.format(mode, period)]) +             np.array(skims['WLK_{0}_WLK_XWAIT__{1}'.format(mode, period)]) +             np.array(skims['WLK_{0}_WLK_WAUX__{1}'.format(mode, period)]) +             np.array(skims['WLK_{0}_WLK_TOTIVT__{1}'.format(mode, period)]))/100
            walk_transit.append(walk_time_skim)

            drive_time_skim = (np.array(skims['DRV_{0}_WLK_DTIM__{1}'.format(mode, period)]) +             np.array(skims['DRV_{0}_WLK_IWAIT__{1}'.format(mode, period)]) +             np.array(skims['DRV_{0}_WLK_XWAIT__{1}'.format(mode, period)]) +             np.array(skims['DRV_{0}_WLK_WAUX__{1}'.format(mode, period)]) +             np.array(skims['DRV_{0}_WLK_TOTIVT__{1}'.format(mode, period)]))/100
            drive_transit.append(drive_time_skim)

        bike_time = np.array(skims['DISTBIKE'])*60/12 #12 miles/hour
        walk_time = np.array(skims['DISTWALK'])*60/3 #3 miles/hour

        period_time_skims = np.stack((driving_skims +                                       [walk_time] +                                       [bike_time] +                                       walk_transit +                                       drive_transit))

        time_skims.append(period_time_skims)

    return np.stack(time_skims)

def driving_skims(skims):
    """
    Return time skims for each mode of transportation.
    Time Period Index:
    - 'EA'= 0
    - 'AM'= 1
    - 'MD'= 2
    - 'PM'= 3
    - 'EV'= 4
    Mode Index:
    -'DRIVE_HVY': 0,
    -'DRIVE_LOC': 1,
    -'DRIVE_EXP': 2,
    -'DRIVE_COM': 3,
    -'DRIVE_LRF': 4,
    - OTHER MODE': 5

    Return:
    - four-dimensional matrix. (time_period, mode, origin, destination)
    """
    periods = ['EA', 'AM', 'MD', 'PM', 'EV']
    transit_modes = ['HVY','LOC','EXP','COM','LRF']

    time_skims = []
    for period in periods:
        driving_access_skims = []
        for mode in transit_modes:
            drive_access_skim = (np.array(skims['DRV_{0}_WLK_DDIST__{1}'.format(mode, period)]))/100
            driving_access_skims.append(drive_access_skim)
            shape = drive_access_skim.shape


        period_time_skims = np.stack(driving_access_skims + [np.zeros(shape)])

        time_skims.append(period_time_skims)

    return np.stack(time_skims)

def add_results_variables(settings, trips, households, persons, skims, land_use):
    # trip_ids = [ 504934577, 1751074894, 1777578817, 1456859106,  603528097,
    #         1976026805,  533494525,  289144125, 1633307365, 1638300725,
    #         1467097413,  157430480, 1940255009,  914568381,  523906186,
    #         1504426761, 1971365534, 1962963501, 1186364457, 1094031193]
    # df = trips[trips.index.isin(trip_ids)].copy()
    df = trips.copy()

    #Skims values
    dist = np.array(skims['DIST'])
    time_skims_final = time_skims(skims)
    driving_access_skims = driving_skims(skims)

    # Mappings
    carb_mode_mapping = settings['carb_mode_mapping']
    mode_index_mapping = settings['mode_index_mapping']
    drv_acc_mode_index_mapping = settings['driving_access_mode_index_mapping']
    commute_mapping = settings['commute_mapping']
    ivt_mapping = settings['ivt_mapping']
    hispanic = settings['hispanic']
    county_mapping = settings['county_mapping']
    area_type = settings['area_type']

    df['dist_miles'] = od_matrix_lookup(df.origin, df.destination, dist)
    df['carb_mode'] = df.trip_mode.replace(carb_mode_mapping)
    df['commute'] = df.primary_purpose.replace(commute_mapping)
    df['period'] = pd.cut(df.depart, (0,5,10,15,19,24), labels = [0,1,2,3,4]).astype(int)
    df['mode_index'] = df.trip_mode.replace(mode_index_mapping)
    df['travel_time'] = od_matrix__time_lookup(df.period, df.mode_index,
                                              df.origin, df.destination,
                                              time_skims_final)
    df['driving_access_mode_index'] = df.trip_mode.replace(drv_acc_mode_index_mapping)
    df['driving_access'] = driving_access_skims[df.period, df.driving_access_mode_index,
                                                df.origin -1, df.destination -1]

    df['VMT'] = df['driving_access']
    df['VMT'] = df.VMT.mask(df.trip_mode.isin(['DRIVEALONEFREE','DRIVEALONEPAY']),
                            df.dist_miles)
    df['VMT'] = df.VMT.mask(df.trip_mode.isin(['SHARED2FREE','SHARED2PAY']),
                            df.dist_miles/2)
    df['VMT'] = df.VMT.mask(df.trip_mode.isin(['SHARED3FREE','SHARED3PAY']),
                            df.dist_miles/3)
    df['VMT'] = df.VMT.mask(df.trip_mode.isin(['TNC_SINGLE']), df.dist_miles)
    df['VMT'] = df.VMT.mask(df.trip_mode.isin(['TNC_SHARED']), df.dist_miles/2.5)
    df['VMT'] = df.VMT.mask(df.trip_mode.isin(['TAXI']), df.dist_miles)
    
    land_use['area_type'] = land_use['area_type'].replace(area_type)

    # Add Trips Variables
    df['c_ivt'] = df['primary_purpose'].replace(ivt_mapping)

    #Add socio-economic variables
    df_trips = df.merge(households[['income','lcm_county_id', 'TAZ']], how = 'left',
            left_on = 'household_id', right_index = True)

    df_trips = df_trips.merge(persons[['race','hispanic','value_of_time']], how = 'left',
                         left_on = 'person_id', right_index = True)
    
    df_trips = df_trips.merge(land_use[['area_type']], how = 'left', 
                             left_on = 'TAZ', right_index = True)
    

#     print('Mean TRIP VOT: {}'.format(df_trips['value_of_time'].mean()))

    df_trips['income_category'] = pd.cut(df_trips.income,
                                         [-np.inf, 80000, 150000, np.inf],
                                         labels = ['low', 'middle','high'])

    df_trips['c_cost'] = (0.60 * df_trips['c_ivt'])/(df_trips['value_of_time'])
    df_trips['cs'] = df_trips['mode_choice_logsum']/(-1 * df_trips['c_cost'].mean())

    df_trips['hispanic'] = df_trips['hispanic'].replace(hispanic)
    df_trips['lcm_county_id'] = df_trips['lcm_county_id'].replace(county_mapping)

    ## Modify persons table
    df_persons = persons.copy()

    df_persons = df_persons.merge(households[['income','lcm_county_id']],
                                  how = 'left',
                                  left_on = 'household_id',
                                  right_index = True)
    
    df_persons = df_persons.merge(land_use[['area_type']], how = 'left', 
                             left_on = 'TAZ', right_index = True)

    df_persons['income_category'] = pd.cut(df_persons.income,
                                         [-np.inf, 80000, 150000, np.inf],
                                         labels = ['low', 'middle','high'])
    df_persons['lcm_county_id'] = df_persons['lcm_county_id'].replace(county_mapping)
    df_persons['hispanic'] = df_persons['hispanic'].replace(hispanic)

#     print('Trips shape, after merging: {}'.format(df_trips.shape))
#     print('Persons shape after merging: {}'.format(df_persons.shape))
#     print('Sum of c_cost: {}'.format(df_trips['c_cost'].sum()))
#     print('Sum of c_ivt: {}'.format(df_trips['c_ivt'].sum()))
#     print('Sum of value of time: {}'.format(df_trips['value_of_time'].sum()))
#     print('Sum of cs: {}'.format(df_trips['cs'].sum()))
    return df_trips, df_persons

############################
## Performance Indicators ##
############################


## VMT ##
#########

def total_vmt(trips):
    logging.info('Calulating total VMT...')
    return float(trips['VMT'].sum())

def vmt_per_capita(trips, persons):
    logging.info('Calulating VMT per capita...')
    n = len(persons)
    total = trips['VMT'].sum()
    return float(total/n)

def vmt_per_capita_income(trips, persons):
    logging.info('Calulating VMT per capita by income...')
    persons_segment = persons.groupby('income_category')['PNUM'].count()
    total_segment = trips.groupby('income_category').agg({'VMT':'sum'})
    kpi = total_segment.div(persons_segment, axis = 0)
    return kpi.to_dict()['VMT']

def vmt_per_capita_race(trips, persons):
    logging.info('Calulating VMT per capita by race...')
    persons_segment = persons.groupby('race')['PNUM'].count()
    total_segment = trips.groupby('race').agg({'VMT':'sum'})
    kpi = total_segment.div(persons_segment, axis = 0)
    return kpi.to_dict()['VMT']

def vmt_per_capita_hispanic(trips, persons):
    logging.info('Calulating VMT per capita by hispanic...')
    persons_segment = persons.groupby('hispanic')['PNUM'].count()
    total_segment = trips.groupby('hispanic').agg({'VMT':'sum'})
    kpi = total_segment.div(persons_segment, axis = 0)
    return kpi.to_dict()['VMT']

def vmt_per_capita_county(trips, persons):
    logging.info('Calulating VMT per capita by county...')
    persons_segment = persons.groupby('lcm_county_id')['PNUM'].count()
    total_segment = trips.groupby('lcm_county_id').agg({'VMT':'sum'})
    kpi = total_segment.div(persons_segment, axis = 0)
    return kpi.to_dict()['VMT']

def vmt_per_capita_urban(trips, persons):
    logging.info('Calulating VMT per capita by urban classification...')
    persons_segment = persons.groupby('area_type')['PNUM'].count()
    total_segment = trips.groupby('area_type').agg({'VMT':'sum'})
    kpi = total_segment.div(persons_segment, axis = 0)
    return kpi.to_dict()['VMT']

## Consumer Surplus ##
######################
def total_consumer_surplus(df):
    logging.info('Calulating consumer surplus...')
    return float(df['cs'].sum())

def consumer_surplus_per_capita(trips, persons):
    logging.info('Calulating average consumer surplus...')
    n = len(persons)
    total = trips['cs'].sum()
    return float(total/n)

def consumer_surplus_per_capita_income(trips, persons):
    logging.info('Calulating average consumer surplus by income...')
    persons_segment = persons.groupby('income_category')['PNUM'].count()
    total_segment = trips.groupby('income_category').agg({'cs':'sum'})
    kpi = total_segment.div(persons_segment, axis = 0)
    return kpi.to_dict()['cs']

def consumer_surplus_per_capita_race(trips, persons):
    logging.info('Calulating average consumer surplus by race...')
    persons_segment = persons.groupby('race')['PNUM'].count()
    total_segment = trips.groupby('race').agg({'cs':'sum'})
    kpi = total_segment.div(persons_segment, axis = 0)
    return kpi.to_dict()['cs']

def consumer_surplus_per_capita_hispanic(trips, persons):
    logging.info('Calulating average consumer surplus by hispanic...')
    persons_segment = persons.groupby('hispanic')['PNUM'].count()
    total_segment = trips.groupby('hispanic').agg({'cs':'sum'})
    kpi = total_segment.div(persons_segment, axis = 0)
    return kpi.to_dict()['cs']

def consumer_surplus_per_capita_county(trips, persons):
    logging.info('Calulating average consumer surplus by county...')
    persons_segment = persons.groupby('lcm_county_id')['PNUM'].count()
    total_segment = trips.groupby('lcm_county_id').agg({'cs':'sum'})
    kpi = total_segment.div(persons_segment, axis = 0)
    return kpi.to_dict()['cs']

def consumer_surplus_per_capita_urban(trips, persons):
    logging.info('Calulating average consumer surplus by county...')
    persons_segment = persons.groupby('area_type')['PNUM'].count()
    total_segment = trips.groupby('area_type').agg({'cs':'sum'})
    kpi = total_segment.div(persons_segment, axis = 0)
    return kpi.to_dict()['cs']


## others ##
############
def transit_ridersip(trips):
    logging.info('Calulating transit ridership...')
    return int(trips.carb_mode.isin(['Public Transit']).sum())

def mode_shares(trips):
    logging.info('Calulating mode shares...')
    ms = trips.carb_mode.value_counts(normalize=True)
    return ms.to_dict()

def mode_shares_trips(trips):
    logging.info('Calulating mode shares trips...')
    ms = trips.carb_mode.value_counts(normalize=False)
    return ms.to_dict()

def average_vehicle_ownership(households):
    logging.info('Calulating vehicle ownership...')
    return float(households.auto_ownership.mean())

def seat_utilization(trips):
    logging.info('Calulating seat utilization...')
    veh_1 = int(trips['trip_mode'].isin(['DRIVEALONEFREE','DRIVEALONEPAY']).sum())
    veh_2 = int(trips['trip_mode'].isin(['SHARED2FREE','SHARED2PAY']).sum())
    veh_3 = int(trips['trip_mode'].isin(['SHARED3FREE','SHARED3PAY']).sum())
    return float((veh_1 + veh_2 + veh_3)/(veh_1 + veh_2/2 + veh_3/3))

def average_traveltime_purpose(trips):
    logging.info('Calulating average travel time by purpose...')
    s = trips.groupby('commute').agg({'travel_time':'mean'})
    return s.to_dict()['travel_time']

def average_traveltime_mode(trips):
    logging.info('Calulating average travel time by mode...')
    s = trips.groupby('carb_mode').agg({'travel_time':'mean'})
    return s.to_dict()['travel_time']

def average_traveltime_income(trips):
    logging.info('Calulating average travel time by income...')
    s = trips.groupby('income_category').agg({'travel_time':'mean'})
    return s.to_dict()['travel_time']

def average_commute_trip_lenght(trips):
    logging.info('Calulating average commute trip lenght...')
    s = trips.groupby('carb_mode').agg({'dist_miles':'mean'})
    return s.to_dict()['dist_miles']

def kpi_results_test(scenario, trips, households, persons, skims, land_use):
    
    mapping = read_yaml('../mapping.yaml')
    
    trips_ , persons_ = add_results_variables(mapping, trips, households, persons, skims, land_use)
    
    dict_results = {}
    dict_results['policy'] = scenario[3:-5]
    dict_results['name'] = scenario

    #Vmt KPIS
    dict_results['total_vmt'] = total_vmt(trips_)
    dict_results['vmt_per_capita'] = vmt_per_capita(trips_, persons_)
    dict_results['vmt_per_capita_income'] = vmt_per_capita_income(trips_, persons_)
    dict_results['vmt_per_capita_race'] = vmt_per_capita_race(trips_, persons_)
    dict_results['vmt_per_capita_hispanic'] = vmt_per_capita_hispanic(trips_, persons_)
    dict_results['vmt_per_capita_county'] = vmt_per_capita_county(trips_, persons_)
    dict_results['vmt_per_capita_urban'] = vmt_per_capita_urban(trips_, persons_)

    #Consumer Surplus KPIs
    dict_results['total_cs'] = total_consumer_surplus(trips_)
    dict_results['cs_per_capita'] = consumer_surplus_per_capita(trips_, persons_)
    dict_results['cs_per_capita_income'] = consumer_surplus_per_capita_income(trips_, persons_)
    dict_results['cs_per_capita_race'] = consumer_surplus_per_capita_race(trips_, persons_)
    dict_results['cs_per_capita_hispanic'] = consumer_surplus_per_capita_hispanic(trips_, persons_)
    dict_results['cs_per_capita_county'] = consumer_surplus_per_capita_county(trips_, persons_)
    dict_results['cs_per_capita_urban'] = consumer_surplus_per_capita_urban(trips_, persons_)

    #Other
    dict_results['transit_ridersip'] = transit_ridersip(trips_)
    dict_results['mode_shares'] = mode_shares(trips_)
    dict_results['mode_shares_trips'] = mode_shares_trips(trips_)
    dict_results['average_vehicle_ownership'] = average_vehicle_ownership(households)
    dict_results['seat_utilization'] = seat_utilization(trips_)
    dict_results['average_traveltime_purpose'] = average_traveltime_purpose(trips_)
    dict_results['average_traveltime_mode'] = average_traveltime_mode(trips_)
    dict_results['average_traveltime_income'] = average_traveltime_income(trips_)
    dict_results['average_commute_trip_lenght'] = average_commute_trip_lenght(trips_)

    return dict_results


# # Re-estimate KPI

# In[ ]:


def rewrite_kpi(scenario):
    
    #Read trips table
    trips_fpath = os.path.join(scenario, "final_trips.csv")
    trips = read_csv(trips_fpath, index_col = 'trip_id', dytpe = {'origin':int, 'destination':int} )
    
    # Read households table
    households_fpath = os.path.join(scenario, "final_households.csv")
    households = read_csv(households_fpath, index_col = 'household_id')
    
    # Read persons Table
    persons_fpath = os.path.join(scenario, "final_persons.csv")
    persons = read_csv(persons_fpath, index_col = 'person_id')
    
    # Read skims
    skims_fpath = f"../skims/{scenario}_skims.omx"
    download_s3(skims_fpath, 'carb-results', f"{scenario}/skims.omx")
    skims = omx.open_file(skims_fpath, mode = 'r')
    
    # Read land use table
    land_use_fpath = os.path.join(scenario, "final_land_use.csv")
    land_use = read_csv(land_use_fpath, index_col = 'TAZ')
    
    # Re-estimate results 
    results = kpi_results_test(scenario, trips, households, persons, skims, land_use)
    
    # Save yaml 
    yaml_fpath = f"updated_kpi_{scenario}.yaml"
    save_yaml(yaml_fpath, results)

    #Upload to S3 Bucket
    s3_name = f"{scenario}/{yaml_fpath}"
    upload_file_s3(yaml_fpath, bucket = 'carb-results', object_name=s3_name)


# In[ ]:


scenarios = pd.Series(os.listdir('../kpis'))
scenarios = scenarios[scenarios.str.contains('.yaml')].str[4:-5]
scenarios = list(scenarios.sort_values())


# In[ ]:


for scenario in scenarios[31:]:
    print(f'Re-estiamting scenario {scenario}')
    start = time.time()
    rewrite_kpi(scenario)
    end = time.time()
    print(f"Time: {end - start:.2f} seconds")
    print("")


# In[ ]:




