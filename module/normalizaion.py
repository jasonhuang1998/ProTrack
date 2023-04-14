def normalization(merged_data):
    for mobile in merged_data:
        df = merged_data[mobile]
        df = changeZero(df)
        df = columnNormalization(df)
        merged_data[mobile] = df
    return merged_data


def changeZero(df):
    column_list = list(df.columns)
    column_list.remove('time')
    for column_name in column_list:
        for i in range(len(df[column_name])):
            if df[column_name][i] == 0:
                df._set_value(i, column_name, -110)
    return df


def columnNormalization(df):
    column_list = list(df.columns)
    column_list.remove('time')
    for column_name in column_list:
        max = df[column_name].max()
        min = df[column_name].min()
        for i in range(len(df[column_name])):
            if (max - min) != 0:
                value = (df[column_name][i] - min)/(max - min)
            else:
                value = max
            df._set_value(i, column_name, value)
    return df