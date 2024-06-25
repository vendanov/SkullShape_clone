





import os
import glob

import pandas as pd
import mysql.connector
import fnmatch
from pathlib import Path


cohort_folder = '/home/castrone/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/Screen20210315c182_J01Rik & Ascl5 & wt'  # CHANGE HERE


cohort_number = cohort_folder.split('/')[-1].split('c')[1].split('_')[0]



all_files = glob.glob(os.path.join(cohort_folder + '/' + 'BMD_TMD', "*.csv"))  ################# CHANGE HERE
df_from_each_file = (pd.read_csv(f, sep=',', header=0) for f in all_files)
df_merged = pd.concat(df_from_each_file, ignore_index=True)

df_merged.drop(df_merged.filter(regex="Unname"), axis=1, inplace=True)
df_merged = df_merged.sort_values(by='CCP_mouseID')
df_merged = df_merged.reset_index(drop=True)
df_merged.rename(
    columns={'CCP_mouseID': 'Identificator', 'CCP_date': 'CCP_DXA_001', 'CCP_metadata': '', 'CCP_DXA_793_001': '',
             'CCP_DXA_004_001': ''}, inplace=True)
df_merged.loc[-1] = ['CCP_mouseID', 'CCP_date', 'CCP_metadata', 'CCP_DXA_793_001', 'CCP_DXA_004_001']
df_merged.index = df_merged.index + 1
df_merged.loc[-1] = ['MouseID', 'Date', 'MetaDatalink', 'TMD', 'BMD']
df_merged.index = df_merged.index + 1
df_merged = df_merged.sort_index()
# print(df_merged)
df_merged.to_excel(cohort_folder + '/' + 'DXA ' + cohort_number + '.xlsx', index=False, index_label=False)

# connect to the DB
conn = mysql.connector.connect(host='london02.img.local', port=3306, database='vestlontest', user='vestlontest_bi',
                               password='vJPeF5T6a83j$w8N')
cursor = conn.cursor()
sql = "SELECT * FROM MOUSE WHERE COHORT_ID ='%s'" % (cohort_number)
cursor.execute(sql)
database = pd.DataFrame(cursor.fetchall(), columns=['DOB', 'ID', 'SACRIFICE', 'LAB_ID', 'COHORT_ID', 'CAGE_ID'])
cursor.close()

# Read the excel file
sheet1_df = pd.read_excel(cohort_folder + '/' + 'DXA ' + cohort_number + '.xlsx')

# transform DF
excel_transf = sheet1_df[['Identificator', 'CCP_DXA_001']][2:]
excel_transf = excel_transf.reset_index(drop=True)
excel_transf = excel_transf.rename(columns={"Identificator": "ID", 'CCP_DXA_001': 'Folder_date'})

# transform database table
database = database.sort_values(by='ID')
database = database[['ID', 'SACRIFICE']]

# find ids that are not in excel table
died = database['ID'][~database['ID'].isin(excel_transf['ID'])].dropna()
died = died.tolist()
for idd in died:
    new_row = {'Identificator': idd, 'CCP_DXA_001': '2021-12-13', 'Unnamed: 2': 'UCT 0245',
               'Unnamed: 3': 'IMPC_PSC_001', 'Unnamed: 4': 'IMPC_PSC_001'}
    sheet1_df = sheet1_df.append(new_row, ignore_index=True)
#
# for i in df_cohort_1176['ID_without'].values:
# sheet1_df.loc[sheet1_df['Identificator'] == i, 'Unnamed: 2'] = 'UCT 0241'

# print(sheet1_df)
# add ids with code
sheet1_df = sheet1_df[2:].sort_values(by='Identificator')
sheet1_df = sheet1_df.reset_index()
sheet1_df.pop('index')
sheet1_df.loc[-1] = ['CCP_mouseID', 'CCP_date', 'CCP_metadata', 'CCP_DXA_793_001', 'CCP_DXA_004_001']
sheet1_df.index = sheet1_df.index + 1
sheet1_df.loc[-1] = ['MouseID', 'Date', 'MetaDatalink', 'TMD', 'BMD']
sheet1_df.index = sheet1_df.index + 1
sheet1_df = sheet1_df.sort_index()
sheet1_df = sheet1_df.rename(columns={"Unnamed: 2": " ", 'Unnamed: 3': '', 'Unnamed: 4': ''})

Python_version = !python - V

# Create DataFrame
if len(df_cohort_1176['ID']) > 0:
    sheet2_df = pd.DataFrame({'MetaData': ['CCP_metadata', 'UCT 0241', 'UCT 0245'],
                              'Date equipment last calibrated': ['CCP_DXA_017_001', excel_transf['Folder_date'][0],
                                                                 excel_transf['Folder_date'][0]],
                              'Operator: Rec': ['CCP_DXA_999_000', 'DXA_SD_009', 'DXA_SD_009'],  # DXA_SD_009
                              'Operator: ROI': ['CCP_DXA_016_001', 'DXA_sfw_009', 'DXA_sfw_009'],
                              'Operator: BMD': ['CCP_DXA_996_000', 'DXA_sfw_009', 'DXA_sfw_009'],
                              'Equipment ID': ['CCP_DXA_011_001', '19G14030', '14G08101'],  # 14G08101
                              'Equipment manufacturer': ['CCP_DXA_012_001', 'Bruker', 'Bruker'],  # Bruker
                              'Equipment model': ['CCP_DXA_013_001', 'SkyScan 1176', 'SkyScan 1278'],
                              'Mouse Status': ['CCP_DXA_014_001', 'Anesthetized', 'Anesthetized'],
                              'Anesthesia': ['CCP_DXA_015_001', 'Zoletil+Xylazine', 'Zoletil+Xylazine'],
                              'DXA_Analytical software producer': ['CCP_DXA_993_001', 'Python Software Foundation',
                                                                   'Python Software Foundation'],
                              'DXA_Analytical software': ['CCP_DXA_992_001', 'Python', 'Python'],
                              'DXA_Analytical software version': ['CCP_DXA_991_001', Python_version[0],
                                                                  Python_version[0]]})
    print(sheet1_df)
else:
    sheet2_df = pd.DataFrame({'MetaData': ['CCP_metadata', 'UCT 0245'],
                              'Date equipment last calibrated': ['CCP_DXA_017_001', excel_transf['Folder_date'][0]],
                              'Operator: Rec': ['CCP_DXA_999_000', 'DXA_SD_009'],  # DXA_SD_009
                              'Operator: ROI': ['CCP_DXA_016_001', 'DXA_sfw_009'],
                              'Operator: BMD': ['CCP_DXA_996_000', 'DXA_sfw_009'],
                              'Equipment ID': ['CCP_DXA_011_001', '19G14030'],  # 14G08101
                              'Equipment manufacturer': ['CCP_DXA_012_001', 'Bruker'],  # Bruker
                              'Equipment model': ['CCP_DXA_013_001', 'SkyScan 1278'],
                              'Mouse Status': ['CCP_DXA_014_001', 'Anesthetized'],
                              'Anesthesia': ['CCP_DXA_015_001', 'Zoletil+Xylazine'],
                              'DXA_Analytical software producer': ['CCP_DXA_993_001', 'Python Software Foundation'],
                              'DXA_Analytical software': ['CCP_DXA_992_001', 'Python'],
                              'DXA_Analytical software version': ['CCP_DXA_991_001', Python_version[0]]})
    print(sheet1_df)

with pd.ExcelWriter(
        '/home/castrone/PRIMUS/data/83_IMPC_DATA/Bioimaging/STATISTICS/_FINISHED_ Tabs_' + '/' + 'DXA ' + cohort_number + '.xlsx') as writer:
    sheet1_df.to_excel(writer, sheet_name="Sheet1", index=False, index_label=False)
    sheet2_df.to_excel(writer, sheet_name="Sheet2", index=False, index_label=False)