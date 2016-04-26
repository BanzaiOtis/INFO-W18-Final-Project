import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from urllib.request import HTTPError
from urllib.request import URLError
import http
import os
import json


####---- Variables ----####
api_key = '2B92C17D184FEB235C00913E20A82629'

states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        # 'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
    #     'GU': 'Guam',
    #    'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        # 'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        # 'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        # 'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        # 'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

####---- Electricity Data Functions ----####

def direct_use_share(row, warnings=True):
    if row['Industrial and commercial generation subtotal'] != 0:
        return row['Direct use'] / row['Industrial and commercial generation subtotal']
    elif row['Total net generation'] != 0:
        if warnings:
            print('Used Tot net gen for', row['state'], row.name)
        if row['Direct use'] == 0:
            if warnings:
                print('... but direct use was 0 also.')
        return row['Direct use'] / row['Total net generation']
    else:
        if warnings:
            print('Could not calculate direct use share for', row['state'], row.name)
        return np.NaN

def direct_use_share_source(row):
    """ Determines whether direct use shares were calculated as a proportion of
    industrial and commmercial net generation subtotal, total net generation, or nothing."""

    if row['Industrial and commercial generation subtotal'] != 0:
        return 'Industrial and commercial generation subtotal'
    elif row['Total net generation'] != 0:
        return 'Total net generation'
    else:
        return np.NaN

def net_imports_share(row, method, warnings=True):
    net_imports = row['net_imports']

    if method == 1:
        divisor = row['Total net generation']
    elif method == 2:
        divisor = row['Total electric industry retail sales']
    elif method == 3:
        divisor = row['Total net generation'] - \
                  row['Direct use'] - \
                  row['Total electric industry retail sales']
    else:
        divisor = 0

    if  divisor != 0:
        return net_imports / divisor
    else:
        if warnings:
            print('Could not calculate net imports share for', row['state'], row.name)
        return np.NaN

def table10_xls_to_df(state, file_dir='./table10/', imports_method=None, warnings=True):
    """ Loads the table10 Excel file for the given state and formats as a dataframe.
    Imports methods are as follows:
    1 -> shares are the ratio of annual imports to annual gen. monthly values are monthly gen * share
    2 -> shares are the ratio of annual imports to retail sales. monthly values are monthly sales * share
    3 -> shares are the ratio of annual imports to annual (gen - direct use - ret sales). monthly values are that sum * share
    """
    file_name = 'sept10' + state.lower() + '.xls'

    df = pd.read_excel(file_dir + file_name, sheetname=0, header=3)
    df = df.T # transposing
    df.columns = df.iloc[0] # set column names to values in first row
    df = df.iloc[1:] # removes first row

    # removing unwanted columns
    columns_to_keep = [8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23]
    keeper_names = [df.columns[i] for i in columns_to_keep]
    df = df[keeper_names]

    # setting date index
    df.index.name = 'date' # setting index name
    df.index = pd.to_datetime(df.index) # converting index to datetime
    df = df.sort_index()

    # adding state column to identify this data when all states are in one dataframe
    df['state'] = state

    # the EIA estimates monthly direct use by calculating annual 'shares': the proportion of annual
    # direct use to the amount of annual net generation from commercial and industrial sectors.
    # Then this proportion is applied to the net commercial/industrial gen in each month to
    # get the direct use estimate for that month. http://www.eia.gov/totalenergy/data/monthly/pdf/sec7.pdf
    df['direct_use_share'] = df.apply(direct_use_share, axis=1, warnings=warnings)
    df['direct_use_share_source'] = df.apply(direct_use_share_source, axis=1)

    # calculating net imports, which is the net amount of import from international and interstate trade
    # then creating share based on the given imports method
    # table 10 lists net interstate trade with exports, so it must be subtracted to obtain net imports
    df['net_imports'] = df['Total international imports'] - df['Total international exports'] - df['Net interstate trade']
    df['net_imports_share'] = df.apply(net_imports_share, axis=1, method=imports_method)

    return df

def table10_merge(state_dict, file_dir="./Electricity_Data/table10/", imports_method=None, warnings=True):
    """ Loads the table10 Excel file for the each state in the given dictionary and merges
    them into a single dataframe.
    Imports methods are as follows:
    1 -> shares are the ratio of annual imports to annual gen. monthly values are monthly gen * share
    2 -> shares are the ratio of annual imports to retail sales. monthly values are monthly sales * share
    3 -> shares are the ratio of annual imports to annual (gen - direct use - ret sales). monthly values are that sum * share
    """
    first = True
    for abrv in state_dict.keys():
        if warnings:
            print(abrv)
        temp = table10_xls_to_df(state=abrv, file_dir=file_dir, imports_method=imports_method, warnings=warnings)
        if first:
            merge = temp.copy()
            first = False
        else:
            merge = merge.append(temp)

    return merge

def query_eia_series(series_id, key):
    """Queries the EIA API for the given series id.
    Returns a nested json.
    """
    api_url = ('http://api.eia.gov/series/?api_key=' + key +
                '&series_id=' + series_id)

    response = urllib.request.urlopen(api_url)
    data = json.loads(response.read().decode('utf-8'))

    # testing if an actual series was returned from the API query
    found = True

    # api queries with an invalid series id will return the below
    # dictionary, so we can test for that.
    if 'data' in data.keys():
        if 'error' in data['data'].keys():
            print('api import error:', data['request']['series_id'])
            found = False

    return data, found

def eia_json_to_dataframe(eia_json, state=None, series_name=None):
    """Converts EIA series json to a pandas dataframe with the
    date as the index.
    """
    data = eia_json['series'][0]['data']

    # setting state and series name
    if not state:
        state = eia_json['series'][0]['geography']
    if not series_name:
        series_name = eia_json['series'][0]['name']

    # then convert to dataframe and set date as datetime
    # not setting as index at this point due to later merging
    df = pd.DataFrame(data, columns=['date', series_name])
    df['date'] = pd.to_datetime(df['date'].astype('str'), format='%Y%m')
    # df.set_index('date', inplace=True)
    # df = df.sort_index()

    # adding state column
    df['state'] = state

    # for validating
    # print("Name:", series_name, "\nUnits:", eia_json['series'][0]['units'])

    return df

def estimate_direct_use(row, annual_df, warnings=True):
    """ Estimates direct use column for the given row from the dataframe of monthly
    electricity data.
    """
    state = row.state
    year = row.date.year
    annual_row = annual_df[(annual_df['state'] == state) & (annual_df.index.year == year)]

    if not annual_row.empty:
        direct_use_share = annual_row['direct_use_share'].item()
        annual_share_source = annual_row['direct_use_share_source'].item()

        if annual_share_source == 'Industrial and commercial generation subtotal':
            monthly_share_source = 'gen_com_ind'
        elif annual_share_source == 'Total net generation':
            monthly_share_source = 'gen_tot'
        else:
            monthly_share_source = None

        if monthly_share_source:
            return row[monthly_share_source] * direct_use_share
        else:
            if warnings:
                print('No monthly share source found for', state, year)
            return np.NaN
    else:
        return np.NaN

def estimate_net_imports(row, annual_df, method=None, warnings=True):
    state = row.state
    year = row.date.year
    annual_row = annual_df[(annual_df['state'] == state) & (annual_df.index.year == year)]

    if not annual_row.empty:
        net_imports_share = annual_row['net_imports_share'].item()

        if method == 1:
            multiplier = row['gen_tot']
        elif method == 2:
            multiplier = row['ret_sales']
        elif method == 3:
            multiplier = row['gen_tot'] - row['direct_use_est'] - row['ret_sales']
        else:
            if warnings:
                print("Error, invalid net imports estimation method for", state, year)
            return np.NaN

        return multiplier * net_imports_share
    else:
        return np.NaN

def estimate_losses(row, warnings=True):
    produced = row['gen_tot'] + row['net_imports_est']
    used = row['ret_sales'] + row['direct_use_est']
    losses = produced - used

    if losses < 0:
        if warnings:
            print('Warning: Negative losses estimated for', row.state, row.date.year)

    return losses

def best_losses_perc(row):
    if np.abs(row['losses2_perc']) <= 1:
        return row['losses2_perc']
    elif np.abs(row['losses_est_perc']) <= 1:
        return row['losses_est_perc']
    else:
        return np.NaN

def download_eia(state_dict, api_key, folder_path):
    for state in state_dict.keys():
        # grabbing json files via API
        gen_tot_id = 'ELEC.GEN.ALL-' + state + '-99.M'
        gen_tot_json = query_eia_series(gen_tot_id, api_key)
        gen_com_id = 'ELEC.GEN.ALL-' + state + '-97.M'
        gen_com_json = query_eia_series(gen_com_id, api_key)
        gen_ind_id = 'ELEC.GEN.ALL-' + state + '-96.M'
        gen_ind_json = query_eia_series(gen_ind_id, api_key)
        ret_sales_id = 'ELEC.SALES.' + state + '-ALL.M'
        ret_sales_json = query_eia_series(ret_sales_id, api_key)

        combined = {gen_tot_id: gen_tot_json,
                    gen_com_id: gen_com_json,
                    gen_ind_id: gen_ind_json,
                    ret_sales_id: ret_sales_json}

        for s_id, s_json in combined.items():
            file_path = folder_path + s_id + '.json'
            with open(file_path, 'w') as outfile:
                json.dump(s_json, outfile)

def load_monthly_electric(state_dict, annual_df, import_method,folder_path='./Electricity_Data/json/', warnings=True):

    ## Loading data from json files
    # initializing series dataframes
    gen_tot = pd.DataFrame()
    gen_com = pd.DataFrame()
    gen_ind = pd.DataFrame()
    ret_sales = pd.DataFrame()

    for state in states.keys():

        gen_tot_id = 'ELEC.GEN.ALL-' + state + '-99.M'
        file_path = folder_path + gen_tot_id + '.json'
        with open(file_path, "rt") as infile:
            gen_tot_json = json.loads(infile.read())

        gen_com_id = 'ELEC.GEN.ALL-' + state + '-97.M'
        file_path = folder_path + gen_com_id + '.json'
        with open(file_path, "rt") as infile:
            gen_com_json = json.loads(infile.read())

        gen_ind_id = 'ELEC.GEN.ALL-' + state + '-96.M'
        file_path = folder_path + gen_ind_id + '.json'
        with open(file_path, "rt") as infile:
            gen_ind_json = json.loads(infile.read())

        ret_sales_id = 'ELEC.SALES.' + state + '-ALL.M'
        file_path = folder_path + ret_sales_id + '.json'
        with open(file_path, "rt") as infile:
            ret_sales_json = json.loads(infile.read())

        # converting json files to dataframes, if they were found
        gen_tot_df = eia_json_to_dataframe(gen_tot_json[0], state=state, series_name='gen_tot')

        if gen_com_json[1] == True:
            gen_com_df = eia_json_to_dataframe(gen_com_json[0], state=state, series_name='gen_com')

        if gen_ind_json[1] == True:
            gen_ind_df = eia_json_to_dataframe(gen_ind_json[0], state=state, series_name='gen_ind')

        ret_sales_df = eia_json_to_dataframe(ret_sales_json[0], state=state, series_name='ret_sales')

        # merging with master dataframe for each series type, if the json was found
        gen_tot = gen_tot.append(gen_tot_df)

        if gen_com_json[1] == True:
            gen_com = gen_com.append(gen_com_df)
        if gen_ind_json[1] == True:
            gen_ind = gen_ind.append(gen_ind_df)

        ret_sales = ret_sales.append(ret_sales_df)

    ## Merging and formatting dataframes into single monthly dataframe
    # merging into a single dataframe
    monthly = gen_tot.copy()
    monthly = pd.merge(monthly, gen_com, how='outer', on=['date','state'])
    monthly = pd.merge(monthly, gen_ind, how='outer', on=['date','state'])
    monthly = pd.merge(monthly, ret_sales, how='outer', on=['date','state'])

    # missing values can be treated as zero in this data set
    monthly = monthly.fillna(0)

    # adding combined commercial/industrial net gen
    monthly['gen_com_ind'] = monthly['gen_com'] + monthly['gen_ind']

    # estimating monthly direct use and net imports from the annual values
    monthly['direct_use_est'] = monthly.apply(estimate_direct_use, axis=1, annual_df=annual_df, warnings=warnings)
    monthly['net_imports_est'] = monthly.apply(estimate_net_imports, axis=1, annual_df=annual_df, method=import_method, warnings=warnings)

    # estimating monthly losses and loss percentages
    monthly['losses_est'] = monthly.apply(estimate_losses, axis=1, warnings=warnings)
    monthly['losses_est_perc'] = monthly['losses_est'] / (monthly['gen_tot'] +
                                                          monthly['net_imports_est'])
    monthly['losses2'] = monthly['gen_tot'] - monthly['direct_use_est'] - monthly['ret_sales']
    monthly['losses2_perc'] = monthly['losses2'] / (monthly['gen_tot'])

    monthly['losses_perc'] = monthly.apply(best_losses_perc, axis=1)

    # setting date as index and sorting
    monthly.set_index('date', inplace=True)
    monthly = monthly.sort_index()

    return monthly

####---- Weather Data Functions ----####
def load_weather(file_path='./source_data/Weather_Data/MonthlyWeatherDataNOAA_US.txt'):
    weather = pd.read_csv(file_path, delimiter=',', skipinitialspace=True)
    weather['date'] = pd.to_datetime(weather['YearMonth'].astype('str'), format='%Y%m')
    weather = weather.drop(['YearMonth', 'StateCode', 'Division', 'Unnamed: 20'], axis=1)

    # Setting index and sorting
    weather.set_index('date', inplace=True)
    weather.sort_index(inplace=True)

    # adding two fields: The annual average temperature and the absolute
    # value of the difference between monthly average and annual.
    annual_tavg = weather.groupby(weather.index.year)['TAVG'].mean()
    weather['ATAVG'] = weather.apply(lambda x: annual_tavg[x.name.year], axis=1)
    weather['DTAVG'] = (weather['TAVG'] - weather['ATAVG']).abs()

    return weather

####---- Satellite Data Functions ----#####

def read_early_satellite(file_path='./GOES_Data/MagneticFluxData_2005-2009.csv'):
    """ Reads minutely satelitte data, then cleans, formats, and aggregates
    it into monthly values.
    """

    df = pd.read_csv(file_path)

    df.drop('Unnamed: 0', inplace=True, axis=1)

    # aggregating average minutely data by minute and coverage
    minute = df.groupby(['time_tag', 'Coverage']).mean()
    minute.reset_index(inplace=True)
    minute.time_tag = pd.to_datetime(minute.time_tag)
    minute.set_index('time_tag', inplace=True)
    minute.sort_index(inplace=True)
    minute.drop('SatID', inplace=True, axis=1)

    minute['yearmonth'] = minute.apply(lambda x: str(x.name.year) +
                                       str(x.name.month), axis=1)

    # Creating version of minutes dataframe with absolute values
    minute_abs = minute.copy()
    minute_abs[['hp', 'ht', 'he', 'hn']] = minute_abs[['hp', 'ht', 'he', 'hn']].abs()

    # Two lists for initializing satellite dataframe
    dates = []
    date_range = range(2005,2010)
    for year in date_range:
        for month in range(1,13):
            dates.extend([str(year) + '-' + str(month) + '-01'] * 2)
    coverage_list = ['East', 'West'] * int(len(dates)/2)

    ## initializing the satellite dataframe for monthly aggregates
    sat = pd.DataFrame(dates)
    sat.columns = ['date']
    sat.date = pd.to_datetime(sat.date)
    sat['yearmonth'] = sat.apply(lambda x: str(x.date.year) +
                                 str(x.date.month), axis=1)
    sat['Coverage'] = coverage_list
    sat.set_index('date', inplace=True)
    sat.sort_index(inplace=True)

    ## adding sums
    temp = minute.groupby(['yearmonth', 'Coverage'])[['hp', 'ht', 'he', 'hn']].sum()
    temp.columns = [(name + '_sum') for name in temp.columns]
    temp.reset_index(inplace=True)
    sat = pd.merge(sat, temp, how='outer', on=['yearmonth', 'Coverage'])

    ## adding abs sums
    temp = minute_abs.groupby(['yearmonth', 'Coverage'])[['hp', 'ht', 'he', 'hn']].sum()
    temp.columns = [(name + '_abs_sum') for name in temp.columns]
    temp.reset_index(inplace=True)
    sat = pd.merge(sat, temp, how='outer', on=['yearmonth', 'Coverage'])

    ## adding means
    temp = minute.groupby(['yearmonth', 'Coverage'])[['hp', 'ht', 'he', 'hn']].mean()
    temp.columns = [(name + '_mean') for name in temp.columns]
    temp.reset_index(inplace=True)
    sat = pd.merge(sat, temp, how='outer', on=['yearmonth', 'Coverage'])

    ## adding abs means
    temp = minute_abs.groupby(['yearmonth', 'Coverage'])[['hp', 'ht', 'he', 'hn']].mean()
    temp.columns = [(name + '_abs_mean') for name in temp.columns]
    temp.reset_index(inplace=True)
    sat = pd.merge(sat, temp, how='outer', on=['yearmonth', 'Coverage'])

    ## adding stds
    temp = minute.groupby(['yearmonth', 'Coverage'])[['hp', 'ht', 'he', 'hn']].std()
    temp.columns = [(name + '_std') for name in temp.columns]
    temp.reset_index(inplace=True)
    sat = pd.merge(sat, temp, how='outer', on=['yearmonth', 'Coverage'])

    ## adding abs stds
    temp = minute_abs.groupby(['yearmonth', 'Coverage'])[['hp', 'ht', 'he', 'hn']].std()
    temp.columns = [(name + '_abs_std') for name in temp.columns]
    temp.reset_index(inplace=True)
    sat = pd.merge(sat, temp, how='outer', on=['yearmonth', 'Coverage'])

    ## adding mins
    temp = minute.groupby(['yearmonth', 'Coverage'])[['hp', 'ht', 'he', 'hn']].min()
    temp.columns = [(name + '_min') for name in temp.columns]
    temp.reset_index(inplace=True)
    sat = pd.merge(sat, temp, how='outer', on=['yearmonth', 'Coverage'])

    ## adding abs mins
    temp = minute_abs.groupby(['yearmonth', 'Coverage'])[['hp', 'ht', 'he', 'hn']].min()
    temp.columns = [(name + '_abs_min') for name in temp.columns]
    temp.reset_index(inplace=True)
    sat = pd.merge(sat, temp, how='outer', on=['yearmonth', 'Coverage'])
    sat.head()

    ## adding maxs
    temp = minute.groupby(['yearmonth', 'Coverage'])[['hp', 'ht', 'he', 'hn']].max()
    temp.columns = [(name + '_max') for name in temp.columns]
    temp.reset_index(inplace=True)
    sat = pd.merge(sat, temp, how='outer', on=['yearmonth', 'Coverage'])

    ## adding abs maxs
    temp = minute_abs.groupby(['yearmonth', 'Coverage'])[['hp', 'ht', 'he', 'hn']].max()
    temp.columns = [(name + '_abs_max') for name in temp.columns]
    temp.reset_index(inplace=True)
    sat = pd.merge(sat, temp, how='outer', on=['yearmonth', 'Coverage'])

    # Setting date index and sorting
    sat['date'] = pd.to_datetime(sat['yearmonth'], format='%Y%m')
    sat.set_index('date', inplace=True)
    sat.sort_index(inplace=True)

    return sat

def verify_url(file_url):
    verified = False
    try:
        c = urllib.request.urlopen(file_url)
        verified = True
    except URLError:
        verified = False

    return verified

def clean_GOES_Data(row):
    for col in df.columns:
        if row[col] == -99999:
            return np.NaN
        else:
            return row[col]

def GOES_csv_to_df(SatId, MaxDay, file_name, FileDir = './GOES_Data/'):
    #print('SatId passed to CSV to DF converstion is %s' % SatId)
    if SatId in range (10,13):
        # Read in file with variable headers

        # ran into memory error.  Trying chunk size
        # http://stackoverflow.com/questions/11622652/large-persistent-dataframe-in-pandas
        tp = pd.read_csv(FileDir + file_name, delimiter=',', skip_blank_lines=False, header=(111 + int(MaxDay)), \
                         parse_dates=['time_tag'], iterator=True, chunksize=1000)
        df = pd.concat(tp, ignore_index=True) # df is DataFrame. If error do list(tp)
        # set index and column names
        df['SatID'] = SatId
        #df.columns = ['time_tag', 'g%shp' % SatId, 'g%she' % SatId, 'g%shn' % SatId, 'g%sht' % SatId]
        #df.index = df['time_tag']
        df.set_index('time_tag', inplace=False)


    elif SatId in range(13,16):
        # Read in file with variable headers
        # ran into memory error.  Trying chunk size
        # http://stackoverflow.com/questions/11622652/large-persistent-dataframe-in-pandas
        tp = pd.read_csv(FileDir + file_name, delimiter=',', skip_blank_lines=False, header=(855 + int(MaxDay)), \
                         parse_dates=['time_tag'], iterator=True, chunksize=1000)
        df = pd.concat(tp, ignore_index=True) # df is DataFrame. If error do list(tp)

        #df = pd.read_csv(FileDir + file_name, delimiter=',', skip_blank_lines=False, header=(855 + int(MaxDay)), \
        #                 parse_dates=['time_tag'])
        # set index and column names
        df.columns = ['time_tag', 'g%s_BX_1_QUAL_FLAG' % SatId, 'g%s_BX_1_NUM_PTS' % SatId, 'g%s_BX_1' % SatId, \
                      'g%s_BY_1_QUAL_FLAG' % SatId, 'g%s_BY_1_NUM_PTS' % SatId, 'g%s_BY_1' % SatId, \
                      'g%s_BZ_1_QUAL_FLAG' % SatId, 'g%s_BZ_1_NUM_PTS' % SatId, 'g%s_BZ_1' % SatId, \
                      'g%s_BXSC_1_QUAL_FLAG' % SatId, 'g%s_BXSC_1_NUM_PTS' % SatId, 'g%s_BXSC_1' % SatId, \
                      'g%s_BYSC_1_QUAL_FLAG' % SatId, 'g%s_BYSC_1_NUM_PTS' % SatId, 'g%s_BYSC_1' % SatId, \
                      'g%s_BZSC_1_QUAL_FLAG' % SatId, 'g%s_BZSC_1_NUM_PTS' % SatId,\
                      'g%s_BZSC_1' % SatId, 'g%s_BTSC_1_QUAL_FLAG' % SatId, 'g%s_BTSC_1_NUM_PTS' % SatId, \
                      'g%s_BTSC_1' % SatId, 'g%s_BX_2_QUAL_FLAG' % SatId, 'g%s_BX_2_NUM_PTS' % SatId, \
                      'g%s_BX_2' % SatId, 'g%s_BY_2_QUAL_FLAG' % SatId, 'g%s_BY_2_NUM_PTS' % SatId, \
                      'g%s_BY_2' % SatId, 'g%s_BZ_2_QUAL_FLAG' % SatId, 'g%s_BZ_2_NUM_PTS' % SatId, \
                      'g%s_BZ_2' % SatId, 'g%s_BXSC_2_QUAL_FLAG' % SatId, 'g%s_BXSC_2_NUM_PTS' % SatId, \
                      'g%s_BXSC_2' % SatId, 'g%s_BYSC_2_QUAL_FLAG' % SatId, 'g%s_BYSC_2_NUM_PTS' % SatId, \
                      'g%s_BYSC_2' % SatId, 'g%s_BZSC_2_QUAL_FLAG' % SatId, 'g%s_BZSC_2_NUM_PTS' % SatId, \
                      'g%s_BZSC_2' % SatId, 'g%s_BTSC_2_QUAL_FLAG' % SatId, 'g%s_BTSC_2_NUM_PTS' % SatId, \
                      'g%s_BTSC_2' % SatId, 'g%s_HP_1_QUAL_FLAG' % SatId, 'g%s_HP_1_NUM_PTS' % SatId, \
                      'g%s_HP_1' % SatId, 'g%s_HE_1_QUAL_FLAG' % SatId, 'g%s_HE_1_NUM_PTS' % SatId, \
                      'g%s_HE_1' % SatId, 'g%s_HN_1_QUAL_FLAG' % SatId, 'g%s_HN_1_NUM_PTS' % SatId, \
                      'g%s_HN_1' % SatId, 'g%s_HT_1_QUAL_FLAG' % SatId, 'g%s_HT_1_NUM_PTS' % SatId, \
                      'g%s_HT_1' % SatId, 'g%s_HP_2_QUAL_FLAG' % SatId, 'g%s_HP_2_NUM_PTS' % SatId, \
                      'g%s_HP_2' % SatId, 'g%s_HE_2_QUAL_FLAG' % SatId, 'g%s_HE_2_NUM_PTS' % SatId, \
                      'g%s_HE_2' % SatId,'g%s_HN_2_QUAL_FLAG' % SatId, 'g%s_HN_2_NUM_PTS' % SatId, \
                      'g%s_HN_2' % SatId, 'g%s_HT_2_QUAL_FLAG' % SatId, 'g%s_HT_2_NUM_PTS' % SatId, 'g%s_HT_2' % SatId]
        #df.index = df['time_tag']
        df.set_index('time_tag', inplace=False)
    else:
        print('error')

    # Set index in place as time
    #df.set_index('time_tag', inplace=True)
    return df
####---- Data Merging Functions ----####
def merge_weather_electric(monthly_df, weather_df, start_year='2001', end_year='2014'):
    monthly_df = monthly_df[start_year:end_year].groupby(monthly_df[start_year:end_year].index).sum()

    # need to recalculate loss percentages, as the sum created in the groupnig
    # above is meaningless
    monthly_df['losses_est_perc'] = monthly_df['losses_est'] / (monthly_df['gen_tot'] +
                                                          monthly_df['net_imports_est'])
    monthly_df['losses2_perc'] = monthly_df['losses2'] / (monthly_df['gen_tot'])

    monthly_df['losses_perc'] = monthly_df.apply(best_losses_perc, axis=1)

    weather_df = weather_df[start_year:end_year]

    merge = pd.merge(monthly_df, weather_df, left_index=True, right_index=True)
    merge.sort_index(inplace=True)
    return merge

####---- Plotting Functions ----####
