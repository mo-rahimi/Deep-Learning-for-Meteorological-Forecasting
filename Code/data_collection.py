
'''In below all data sets are collected to create the final data set for the CW2'''
import pandas as pd
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_rows", None)  # Show all rows

df_tem_max = pd.read_csv("data_sets/Temperature/CLMMAXT_KP_.csv")
df_tem_min = pd.read_csv("data_sets/Temperature/CLMMINT_KP_.csv")
df_tem_mean = pd.read_csv("data_sets/Temperature/CLMTEMP_KP_.csv")
df_cloud = pd.read_csv("data_sets/Amount_of_Cloud/daily_HKO_CLD_ALL.csv")
df_evaporation = pd.read_csv("data_sets/Evaporation/daily_KP_EVAP_ALL.csv")
df_sunshine = pd.read_csv("data_sets/Bright Sunshine/daily_KP_SUN_ALL.csv")
df_humidity = pd.read_csv("data_sets/humidity/daily_KP_RH_ALL.csv")
df_global_solar_radiation = pd.read_csv("data_sets/Global Solar Radiation/daily_KP_GSR_ALL.csv")
df_mean_pressure = pd.read_csv("data_sets/Mean Pressure /daily_HKO_MSLP_ALL.csv")
df_wind_direction = pd.read_csv("data_sets/Prevailing Wind Direction/daily_KP_PDIR_ALL.csv")
df_wind_speed = pd.read_csv("data_sets/Wind Speed/daily_KP_WSPD_ALL.csv")
df_rainfall= pd.read_csv("data_sets/Rainfall/daily_KP_RF_ALL.csv")
df_UV_max = pd.read_csv("data_sets/UV/daily_KP_MAXUV_ALL.csv")
df_UV_mean = pd.read_csv("data_sets/UV/daily_KP_UV_ALL.csv")

# List contain all data frames' names
dataframes = [df_tem_max, df_tem_min, df_tem_mean, df_cloud, df_evaporation,
              df_sunshine, df_humidity, df_global_solar_radiation, df_mean_pressure,
              df_wind_direction, df_wind_speed, df_rainfall, df_UV_max, df_UV_mean]


def print_head_tail(dataframes):
    for i, df in enumerate(dataframes):
        print(f"DataFrame {i + 1}:")
        print("Head:")
        print(df.head())
        print("\nTail:")
        print(df.tail())
        print("\n" + "-" * 50 + "\n")

print_head_tail(dataframes)   # To have a understanding of data frames



# So we do not need column  'completeness', and I will dellete that
def remove_column(df, column_to_remove):    # Function to remove a column from a dataframe
    if column_to_remove in df.columns:
        df = df.drop(column_to_remove, axis=1, inplace=True)
    return df

# Iterate through the dataframes and remove the specified column
column_to_remove = 'completeness'
for i, df in enumerate(dataframes):
    dataframes[i] = remove_column(df, column_to_remove)



#To combine the "year", "month", and "day" columns into a single "date" column with a datetime type.
#This function checks if the "year", "month", and "day" columns exist in each DataFrame. If they do, it creates a new
#"date" column by combining these columns using `pd.to_datetime()`, which converts the combined columns into a datetime type.
# Then, it removes the original "year", "month", and "day" columns using `drop()`, and sets the "date" column as the index.

def process_dataframe(df):
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df.drop(columns=['Year', 'Month', 'Day'], axis=1)
    df = df.iloc[:, [1, 0]]
    df[df.columns[1]] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    return df

'''
I got errors, seems the date in some rows are not correct, actually the problem is in one of the data frames a month have more than 
actual rows, like when a month is 29 days but thre are 30 days rows for that, and I have to find this data frame and 
correct the issue.
ince there are 14 data sets so we have over 300,000 rows, I think it's better to create the column date for each data
frame one by one.'''

df_tem_min = process_dataframe(df_tem_min)
df_tem_max = process_dataframe(df_tem_max)
df_tem_mean = process_dataframe(df_tem_mean)
df_cloud = process_dataframe(df_cloud)
df_evaporation= process_dataframe(df_evaporation)
df_sunshine= process_dataframe(df_sunshine)
df_humidity= process_dataframe(df_humidity)
df_global_solar_radiation= process_dataframe(df_global_solar_radiation)
df_wind_direction= process_dataframe(df_wind_direction)
df_wind_speed = process_dataframe(df_wind_speed)
df_rainfall= process_dataframe(df_rainfall)
df_UV_max = process_dataframe(df_UV_max)
df_UV_mean = process_dataframe(df_UV_mean)
# df_mean_pressure= process_dataframe(df_mean_pressure)

# I realized that data frame "df_mean_pressure" has the problem, below I will try to remove the early month as possibly
# they have the problems
df_mean_pressure = df_mean_pressure.drop(range(0, 21490))
df_mean_pressure['Date'] = pd.to_datetime(df_mean_pressure[['Year', 'Month', 'Day']])

df_mean_pressure= df_mean_pressure.drop(columns=['Year', 'Month', 'Day'], axis=1)
df_mean_pressure = df_mean_pressure.iloc[:, [1,0]]
df_mean_pressure.iloc[:, 1] = pd.to_numeric(df_mean_pressure.iloc[:, 1], errors='coerce')

df_mean_pressure = df_mean_pressure.reset_index(drop=True)
print(" The head of df_mean_pressure is:")
print(df_mean_pressure.head())

dataframes = [df_tem_max, df_tem_min, df_tem_mean, df_cloud, df_evaporation,
              df_sunshine, df_humidity, df_global_solar_radiation, df_mean_pressure,
              df_wind_direction, df_wind_speed, df_rainfall, df_UV_max, df_UV_mean]
print("The 12")
print(df_UV_max.head())



for i, df in enumerate(dataframes):
    if "Date" in df.columns:
        index = df.loc[df["Date"] == '2000-01-01']
        index = index.index[0]
        df.drop(range(0, index), inplace=True)
    else:
        print(f"Dataframe {i} does not have a 'Date' column")

for i, df in enumerate(dataframes):
    index = df.loc[df["Date"] == '2000-01-01']
    # print(index)
    index = index.index[0]
    df = df.drop(range(0, index), inplace=True)
for i, df in enumerate(dataframes):
    df = df.reset_index(drop=True, inplace=True)
all_dataframes = [df.set_index('Date') for df in dataframes]
final_df = pd.concat(all_dataframes, axis=1)
print("The shape of final data frame is :", final_df.shape)
print(final_df.head())

nan_counts = final_df.isna().sum()
print(nan_counts)

for column in final_df.columns:
    # Create a mask for missing values in the column
    mask = final_df[column].isna()

    # Calculate the mean of the before and after values
    before_mean = final_df[column].shift(1).fillna(0)
    after_mean = final_df[column].shift(-1).fillna(0)
    mean_values = (before_mean + after_mean) / 2

    # Impute the missing values with the mean values
    final_df[column] = final_df[column].fillna(mean_values)

'''
This code will iterate over each column in the data frame df. It creates a mask to identify the missing values in each column. 
Then, it calculates the mean of the previous and next values using the shift() function and fills any missing values in
 the mean calculation with zero. Finally, it imputes the missing values in each column with the calculated mean values.
'''

final_df.to_csv("Weather_HK_10.csv")

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_rows", None)  # Show all rows
df = pd.read_csv("Weather_HK_00.csv")
print("Shape of df is :", df.shape)
print(df.head())


