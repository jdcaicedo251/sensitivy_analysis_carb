import os
import sys
import pandas as pd 
import numpy as np 

import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns
import yaml
import re

import openmatrix as omx

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

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
            walk_time_skim = (np.array(skims['WLK_{0}_WLK_WAIT__{1}'.format(mode, period)]) +\
             np.array(skims['WLK_{0}_WLK_IWAIT__{1}'.format(mode, period)]) +\
             np.array(skims['WLK_{0}_WLK_XWAIT__{1}'.format(mode, period)]) +\
             np.array(skims['WLK_{0}_WLK_WAUX__{1}'.format(mode, period)]) +\
             np.array(skims['WLK_{0}_WLK_TOTIVT__{1}'.format(mode, period)]))/100
            walk_transit.append(walk_time_skim)

            drive_time_skim = (np.array(skims['DRV_{0}_WLK_DTIM__{1}'.format(mode, period)]) +\
             np.array(skims['DRV_{0}_WLK_IWAIT__{1}'.format(mode, period)]) +\
             np.array(skims['DRV_{0}_WLK_XWAIT__{1}'.format(mode, period)]) +\
             np.array(skims['DRV_{0}_WLK_WAUX__{1}'.format(mode, period)]) +\
             np.array(skims['DRV_{0}_WLK_TOTIVT__{1}'.format(mode, period)]))/100
            drive_transit.append(drive_time_skim)

        bike_time = np.array(skims['DISTBIKE'])*60/12 #12 miles/hour
        walk_time = np.array(skims['DISTWALK'])*60/3 #3 miles/hour
        
        period_time_skims = np.stack((driving_skims + \
                                      [walk_time] + \
                                      [bike_time] + \
                                      walk_transit + \
                                      drive_transit))
        
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

def add_results_variables(settings, trips, households, skims):
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
    
    df_income = df.merge(households[['income']], how = 'left', 
            left_on = 'household_id', right_index = True)

    df_income['income_category'] = pd.cut(df_income.income, 
                               [-np.inf, 80000, 150000, np.inf], 
                               labels = ['low', 'middle','high'])
    
    return df_income

############################
## Performance Indicators ##
############################

def average_vehicle_ownership(households):
    return households.cars.mean() ##FIX ME: This should be 'auto_ownership'

def average_commute_trip_lenght(trips):
    return trips.groupby('carb_mode').agg({'dist_miles':'mean'})

def average_traveltime_purpose(trips):
    return trips.groupby('commute').agg({'travel_time':'mean'})

def average_traveltime_mode(trips):
    return trips.groupby('carb_mode').agg({'travel_time':'mean'})

def average_traveltime_income(trips):
    return trips.groupby('income_category').agg({'travel_time':'mean'})

def mode_shares(trips):
    return trips.carb_mode.value_counts(normalize=True)

def seat_utilization(trips):
    veh_1 = int(trips['trip_mode'].isin(['DRIVEALONEFREE','DRIVEALONEPAY']).sum())
    veh_2 = int(trips['trip_mode'].isin(['SHARED2FREE','SHARED2PAY']).sum())
    veh_3 = int(trips['trip_mode'].isin(['SHARED3FREE','SHARED3PAY']).sum())      
    return float((veh_1 + veh_2 + veh_3)/(veh_1 + veh_2/2 + veh_3/3))

def transit_ridersip(trips):
    return int(trips.carb_mode.isin(['Public Transit']).sum())
               
def total_vmt(trips):
    return float(trips['VMT'].sum())

def vmt_per_capita(trips, persons):
    population = persons.shape[0]
    return float(total_vmt(trips)/population)

def get_scenario_results(policy, scenario, scenario_id):
    logging.info('Saving policy scenario resutls')
    
    #Important tables
    households = pd.read_csv('tmp/{}/households.csv'.format(scenario), index_col = 'household_id')
    persons = pd.read_csv('tmp/{}/persons.csv'.format(scenario), index_col = 'person_id')
    trips = pd.read_csv('tmp/{}/trips.csv'.format(scenario), index_col = 'trip_id')
    skims = omx.open_file('tmp/{}/skims.omx'.format(scenario), 'r')
    mapping = read_yaml('mapping.yaml')
    
    df = add_results_variables(mapping, trips, households, skims)
    
    dict_results = {}
    dict_results['policy'] = policy
    dict_results['name'] = scenario_id
    dict_results['household_vehicle_ownership'] = float(average_vehicle_ownership(households))
    dict_results['average_trip_length'] = average_commute_trip_lenght(df).to_dict()['dist_miles']
    dict_results['average_travel_time_purpose'] = average_traveltime_purpose(df).to_dict()['travel_time']
    dict_results['average_travel_time_mode'] = average_traveltime_mode(df).to_dict()['travel_time']
    dict_results['average_travel_time_income'] = average_traveltime_income(df).to_dict()['travel_time']
    dict_results['mode_shares'] = mode_shares(df).to_dict()
    dict_results['seat_utilization'] = seat_utilization(df)
    dict_results['transit_ridership'] = transit_ridersip(df)
    dict_results['total_vmt'] = total_vmt(df)
    dict_results['vmt_per_capita'] = vmt_per_capita(df, persons)
    
    skims.close()
    
    return dict_results