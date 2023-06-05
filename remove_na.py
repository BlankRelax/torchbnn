from functions import functions

df=functions.read_data('H:\\Datasets\\','UK-HPI-full-file-2022-01.csv')
print("df shape before na cut: ", df.shape)
num_entries = df.shape[0]
print('number of entries in pre cut dataset: ', num_entries)
columns = functions.get_columns(df)
print(columns)
print('print number of columns before cut: ', columns.shape[0]) # number of columns before removing columns

frac_a =[]
count_a=[]
for frac in range(0,11):
    frac=frac*0.1
    frac_a.append(frac)
    count=0
    for col in columns:
        x = df[col].isna().sum()
        if x>=(num_entries*frac):
            count+=1
    count_a.append(count)
functions.plot(frac_a, count_a,'Fraction of na values', 'Number of variables', 'Number of variables against fraction of missing values')
"""
from this we conclude to use a cut off value of 0.6 for deciding what variables to remove

we remove variables that have greater than 10% of there values missing
"""
columns_drop =[]
for col in columns:
    x = df[col].isna().sum()
    if x >= (num_entries * 0.1):
        columns_drop.append(col)
print('Number of columns to drop', len(columns_drop))
df=df.drop(columns=columns_drop)
df=df.dropna()
print("df shape after na cut: ", df.shape)

df=df.drop(columns=['Date', 'AreaCode'])
functions.write_df(df, 'H:\\Datasets\\','UK-HPI-full-file-2022-01_clean.csv' )






