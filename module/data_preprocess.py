import numpy as np
import pandas as pd
import datetime
from module.log import printLog
from itertools import combinations
from numpy.linalg import norm
from module.normalizaion import normalization

def dataPreprocess(config):
    wireless_data = getWirelessData(config)
    wireless_data = normalization(wireless_data)
    wirelessData2Dataset(wireless_data, config)
    

def wirelessData2Dataset(wireless_data, config):
    time_window = config['data_preprocess']['time_window']
    df = createDF2()
    mobile_list = wireless_data.keys()
    mobile_combination = list(combinations(mobile_list, 2))
    for combination in mobile_combination:
        dev1_data = readData(combination[0], wireless_data)
        dev2_data = readData(combination[1], wireless_data)

        for sniffer_id in dev1_data.columns:
            for start_id in range(10, len(dev1_data.index) - 10 - time_window, time_window):
                row_list = []
                row_list.append(combination[0] + '_' + combination[1])
                for shift in range(-10, 11, 1):
                    index = start_id + shift
                    row_list.append(innerProduct(dev1_data[start_id : start_id + time_window][sniffer_id], dev2_data[index : index + time_window][sniffer_id]))
                row_list.append(getRelation(combination[0], combination[1], config))
                df.loc[len(df.index)] = row_list
    df.to_csv('output.csv', sep = ',', index = False)

    return


def getRelation(dev1, dev2, config):
    for relation in config['relation']:
        if dev1 == relation[1] and dev2 == relation[2]:
            return relation[0]


def innerProduct(data1, data2):
    data1 = data1.to_numpy()
    data2 = data2.to_numpy()
    # print(data1, data2)
    if norm(data1)*norm(data2) != 0:
        ans = np.dot(data1, data2) / (norm(data1)*norm(data2))
    else:
        ans = 0
    return ans


def readData(mobile_name, wireless_data):
    df = wireless_data[mobile_name]
    df = df.drop('time', axis=1)
    return df

def createDF2():
    column = ['pair']
    for shift in range(-10, 11, 1):
        if shift == 0:
            column.append('CosSim_ble')
        else:
            column.append('CosSim_ble_shift' + str(shift))
    column.append('State')
    df = pd.DataFrame(columns = column)
    return df


def getWirelessData(config):
    file_list = []
    for file_name in config['sniffer_list']:
        file_list.append(getSnifferFile(file_name, config))
    df_dict = {}
    for mobile_name in config['mobile_list']:
        printLog("Start preprocessing " + mobile_name + ".")
        df = createDF1(config = config)
        for experiment_nummber in range(len(config['end_time'])):
            df1 = createDF1(config = config)
            df1 = addRow(config, df1 , config['mobile_list'][mobile_name], file_list, experiment_nummber)
            df = pd.concat([df, df1])
        df = df.reset_index()
        df = df.drop('index', axis=1)
        df_dict[mobile_name] = df
    return df_dict


def getSnifferFile(wireless_file_name, config):
    df = pd.read_csv('rawdata/' + str(config['date']) + '/' + wireless_file_name + '.csv', parse_dates=[0])
    df.columns = ['time','mac','type','rssi','uuid']
    df = df.drop('mac', axis=1)
    df = df.drop('type', axis=1)
    df = df.astype({'time':'string'})
    df = df.astype({'uuid':'string'})
    for i in range(len(df.index)):
        df._set_value(i, 'time', str(df['time'][i])[:-7])
        df._set_value(i, 'time', str(df['time'][i])[11:])
    return df

def createDF1(config):
    column = ['time']
    column.extend(config['sniffer_list'])
    df = pd.DataFrame(columns = column)
    return df


def addRow(config, df, uuid, file_list, experiment_number):
    for i in range(getRowNum(config, experiment_number)):
        df.loc[len(df.index)] = getRowData(i, config, uuid, file_list, experiment_number)
    return df


def getRowNum(config, experiment_number):
    start_time = string2datetime(config['start_time'][experiment_number]).time()
    end_time = string2datetime(config['end_time'][experiment_number]).time()
    duration = datetime.datetime.combine(datetime.date.min, end_time) - datetime.datetime.combine(datetime.date.min, start_time)
    return duration.seconds


def getRowData(second, config, uuid, file_list, experiment_number):
    ans = []
    time = string2datetime(config['start_time'][experiment_number]) + datetime.timedelta(0, second)
    ans.append(str(time)[11:])
    for file in range(len(config['sniffer_list'])):
        ans.append(getRSSI(time, file_list[file], uuid))
    return ans


def getRSSI(time, raw_data, uuid):
    raw_data = raw_data.loc[raw_data['time'] == str(time)[11:]]
    raw_data = raw_data.loc[raw_data['uuid'] == uuid]
    sum = raw_data['rssi'].sum()
    frequency = raw_data.shape[0]
    if sum == 0:
        return 0
    else:
        return sum / frequency
    

def string2datetime(time):
    date = '2023/01/06 '
    format = '%Y/%m/%d %H:%M:%S'
    output = datetime.datetime.strptime(date + time, format)
    return output


