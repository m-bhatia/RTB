#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

def process_log_data(log_format_sheet, csv_path):
    log_format_df = pd.read_excel(parent_directory + '/data/LogFormat 1.8.xlsx', sheet_name=log_format_sheet)
    log_data_df = pd.read_csv(csv_path, encoding='ISO-8859-1', header=None, names=log_format_df['Field Name'], low_memory=False)
    
    for column, data_type in zip(log_format_df['Field Name'], log_format_df['Data Type']):
        if data_type == 'TimeStamp':
            log_data_df[column] = pd.to_datetime(log_data_df[column])
        elif data_type == 'int':
            log_data_df[column] = log_data_df[column].astype(int)
        else:
            log_data_df[column] = log_data_df[column].astype(object)
    
    return log_data_df

