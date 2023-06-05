from functions import functions

df=functions.read_data('H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv')
df=functions.to_categorical(df,['RegionName'])
columns=functions.get_columns(df)
print(df['RegionName'])
target = 'AveragePrice'

columns_to_drop=[]
r_array=functions.get_pearson_coeff(df, target)
print(r_array)
for i in range(0,len(r_array)):
    if r_array[i]<0.6 and r_array[i]>-0.6:
        columns_to_drop.append(columns[i])
print(columns_to_drop)

df=df.drop(columns=columns_to_drop)
df = df.drop_duplicates()
functions.write_df(df, 'H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv' )

