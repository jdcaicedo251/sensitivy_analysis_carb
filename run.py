import os
import sys
import boto3
import logging
import pandas as pd
from zipfile import ZipFile

import kpi

s3_client = boto3.client('s3')

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_s3(local_file_name,s3_bucket,s3_object_key):
    """
    reference:
    https://stackoverflow.com/questions/41827963/
    track-download-progress-of-s3-file-using-boto3-and-callbacks
    """

    meta_data = s3_client.head_object(Bucket=s3_bucket, Key=s3_object_key)
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
        s3_client.download_fileobj(s3_bucket, s3_object_key, f, Callback=progress)


def download_kpis(scenario):
    local_file_name = "kpis/kpi_{}.yaml".format(scenario)
    s3_bucket = 'carb-results'
    s3_object_key = "{}/kpi_{}.yaml".format(scenario, scenario)

    download_s3(local_file_name, s3_bucket, s3_object_key)


def download_data(scenario):
    """
    Download results (ActivitySim and Skims) of scenario in a tmp folder.

    Parameters:
    -------------
    - scenario: str. scenario name

    Returns:
    --------
    None

    """
    data_exist = os.path.isdir('tmp/{}'.format(scenario))

    if data_exist:
        pass

    else:
        s3_bucket = 'carb-results'

        #Download Asim results
        asim_local_file = "tmp/{}_asim_output.zip".format(scenario)
        asim_s3_object_key = "{}/asim_outputs_2022.zip".format(scenario)
        download_s3(asim_local_file, s3_bucket , asim_s3_object_key)

        #Unzip Asim Results
        with ZipFile(asim_local_file, 'r') as zipObj:
            zipObj.extractall('tmp')

         #Download Skims
        skims_local_file = "tmp/{}/skims.omx".format(scenario)
        skims_s3_object_key = "{}/skims.omx".format(scenario)
        download_s3(skims_local_file, s3_bucket , skims_s3_object_key)

    return None


def delete_data(scenario):
    """
    Deletes data of the give scenario.

    Parameters:
    ------------
    - scenario: str. scenario name.
    """
    if (scenario == '01_base_000') or (scenario == 'ex_1'):
        pass
    else:
        os.rmdir('tmp/{}'.format(scenario))
        os.remove("tmp/{}_asim_output.zip".format(scenario))
    return None

def add_scenario_changes(df,policy_changes):
    """
    Adds the scenario_id percentual change columns df

    Parameters:
    ------------
    df: pandas DataFrame. Policy resutls
    policy_changes. dict. Dictionary with percental
                          change by policy and scenario.

    Returns:
    --------
    df with <scenario_ids>_%change columns.

    """
    changes = pd.DataFrame(policy_changes).T
    changes.index.set_names('policy', inplace = True)
    return df.join(changes, how = 'outer',lsuffix='_metric', rsuffix="_%change")


def scenario_elasticities(df):
    """
    Estimates scenario elasticity.
    """
    metrics = df[df.columns[df.columns.str.contains('_metric')]]
    change = df[df.columns[df.columns.str.contains('_%change')]].values
    baseline = df['base_line']

    elasticity = (metrics.div(baseline, axis = 0) - 1).div(change, axis = 'columns')
    elasticity.columns = elasticity.columns.str[:11]
    elasticity = elasticity.add_suffix('elasticity')
    return df.join(elasticity)


def mean_elasticity(df):
    """
    Estimates the mean elasticity
    """
    scenario_elasticities = df[df.columns[df.columns.str.contains('_elasticity')]]
    mean = scenario_elasticities.mean(axis = 1)
    df['mean_elasticity'] = mean
    return df


def common_entries(dcts):
    """
    Zip function for dicts
    Reference: https://stackoverflow.com/questions/16458340/python-equivalent-of-zip-for-dictionaries
    Change code to return a dictionary instead, and input a list of dicts.
    """
    if not dcts:
        return
    dict_ = {}
    for i in set(dcts[0]).intersection(*dcts[1:]):
        dict_[i] = tuple(d[i] for d in dcts)
    return dict_


def kpis_scenario(policy, scenario, scenario_id):
    """
    Computes the kpi for the given policy, scenario and scenario_id

    Parameters:
    - policy: str. policy name
    - scenario: str. scenario name
    - scenario_id: str. scenario id

    Returns:
    --------
    dict. dict of KPIs
    """

    logger.info('Estimating KPIs for scenario: {}'.format(scenario))
    # try:
    download_kpis(scenario)
    # except:
        # logger.info('Policy {} not found'.format(scenario))
        # pass

    results_exist = os.path.isfile('kpis/kpi_{}.yaml'.format(scenario))
    if results_exist:
        metrics = kpi.read_yaml('kpis/kpi_{}.yaml'.format(scenario))
        metrics['policy'] = policy
        metrics['scenario_id'] = scenario_id

    # else:
    #     download_data(scenario)
    #     metrics = kpi.get_scenario_results(policy, scenario, scenario_id)
    #     kpi.save_yaml('kpis/{}.yaml'.format(scenario), metrics)

    kpis = list(set(metrics.keys()) - set(['policy', 'name', 'scenario_id']))
    dfs_dict = {}

    for i in kpis:
        try:
            n_categories = len(metrics[i])
            categories = metrics[i].keys()
            baselines = metrics[i].values()

        except TypeError:
            n_categories = 1
            categories = 'none'
            baselines = [metrics[i]]

        scenario_name = metrics['scenario_id']

        df = pd.DataFrame({'policy': [metrics['policy']] * n_categories ,
                           'category': categories,
                           '{}'.format(scenario_name): baselines})

        df = df.set_index(['policy','category'])
        dfs_dict[i] = df

#     delete_data(scenario)
    return dfs_dict

def save_df(name, df):
    """
    Saves dataframe
    """
    df.to_csv('kpis/summary/{}.csv'.format(name))


if __name__ == '__main__':

    policy_scenarios= kpi.read_yaml('policies.yaml')
    policy_changes = kpi.read_yaml('policy_changes.yaml')

#     policy_scenarios = {'policy_one': {'base_line':'ex_1',
#                                        'scenario_1':'ex_2',
#                                        'scenario_2':'ex_3'},
#                         'policy_two': {'base_line':'ex_1',
#                                        'scenario_1':'ex_4',
#                                        'scenario_2':'ex_5'}}

#     policy_changes = {'policy_one': {'scenario_1':0.25,
#                                      'scenario_2':-0.25},
#                       'policy_two': {'scenario_1':0.1,
#                                      'scenario_2':0.25}}



    metrics_list = []
    for policy, scenarios in policy_scenarios.items():

        scenario_list = [kpis_scenario(policy,s,s_id) for s_id,s in scenarios.items()]
        iterable = common_entries(scenario_list)
        scenario_list = {k:pd.concat(v, axis = 1) for k, v in iterable.items()}
        metrics_list.append(scenario_list)

    iterable = common_entries(metrics_list)
    dfs = {k:pd.concat(v, axis = 0) for k, v in iterable.items()}
    dfs = {k:add_scenario_changes(v, policy_changes) for k, v in dfs.items()}
    dfs = {k:scenario_elasticities(v) for k, v in dfs.items()}
    dfs = {k:mean_elasticity(v) for k, v in dfs.items()}
    [save_df(name, df) for name, df in dfs.items()]
