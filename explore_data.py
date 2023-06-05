from functions import functions
##################################PART 1################################## getting correlated variables

# df=functions.read_data('H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv')
# df=functions.to_categorical(df,['RegionName'])
# columns=functions.get_columns(df)
# print(df['RegionName'])
target = 'AveragePrice'
#
# columns_to_drop=[]
# r_array=functions.get_pearson_coeff(df, target)
# print(r_array)
# for i in range(0,len(r_array)):
#     if r_array[i]<0.6 and r_array[i]>-0.6:
#         columns_to_drop.append(columns[i])
# print(columns_to_drop)
#
# df=df.drop(columns=columns_to_drop)
# df = df.drop_duplicates()
# df = functions.min_max_scale(df)
# functions.write_df(df, 'H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv' )

##################################PART 2################################## standardising using min max
# df=functions.read_data('H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv')
# columns=functions.get_columns(df)
# df = functions.min_max_scale(df, columns)
# print(df)
# r_array=functions.get_pearson_coeff(df, target)
# print(r_array)
# functions.write_df(df, 'H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv')

##################################PART 3################################## outlier imputation
df=functions.read_data('H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv')
print(df.shape)
columns=functions.get_columns(df)
functions.make_boxplot(df, columns, (3,3))
for col in columns:
    ro = functions.remove_outliers(df, colname=col)
    ro.find_outliers(sd=5)
ro.write_new('H:\\Datasets\\','UK-HPI-full-file-2022-01_clean_temp.csv')
##################################PART 4################################## check new data
df=functions.read_data('H:\\Datasets\\','UK-HPI-full-file-2022-01_clean_temp.csv')
columns=functions.get_columns(df)
functions.make_boxplot(df, columns, (3,3))
r_array=functions.get_pearson_coeff(df, target)
print(r_array)

