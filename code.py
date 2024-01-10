import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
#from sklearn.linear_model import LinearRegression
sns.set()

df= pd.read_csv("/home/hieu/Downloads/_data_science__ML_data_minning/intro2ds_capstone_project/combined_file.csv",thousands=',',skipinitialspace=True)
df.head()
df.info()

df1 = df[df.DienTich.notnull()]
df1 = df1.reset_index()
for i in range(0,len(df1.DienTich)):
    df1.DienTich[i]= df1.DienTich[i].split(' ')[0]

## Em sẽ chỉ lấy những data có dữ liệu về phòng ngủ 
df2 = df1[df1.Phongngu.notnull()]
for i in range(0,len(df2.Phongngu)):
    if 'nhiều hơn' in df2.Phongngu[i]:
        df2.Phongngu[i]= float(df2.Phongngu[i].split(' ')[2])+1
    else:
        df2.Phongngu[i]= df2.Phongngu[i].split(' ')[0]

## Em sẽ chỉ lấy những data có dữ liệu về phòng tắm
df3 = df2[df2.PhongTam.notnull()]
df3 = df3.reset_index()
for i in range(0,len(df3.PhongTam)):
    if 'Nhiều hơn' in df3.PhongTam[i]:
        df3.PhongTam[i]= float(df3.PhongTam[i].split(' ')[2])+1
    else:
        df3.PhongTam[i]= df3.PhongTam[i].split(' ')[0]
        
del df3['level_0']
del df3['index']


df4 = df3[df3.Loai.notnull()]
for i in range(0,len(df4.Gia)):
    if 'GIÁ TỐT' in df4.Gia[i]:
        df4.Gia[i] = df4.Gia[i].split('\n')[0]
    else:
        df4.Gia[i] = df4.Gia[i].split('-')[0]

for i in range(0,len(df4.Gia)):
    if 'tỷ' in df4.Gia[i]:
        price = df4.Gia[i].split(' ')[0]
        price = price.replace(',','.')
        df4.Gia[i] = round(float(price)*1000000000,1)
    elif 'triệu' in df4.Gia[i] :
        price = df4.Gia[i].split(' ')[0]
        price = price.replace(',','.')
        df4.Gia[i] = round(float(price)*1000000,1)
df4 = df4.drop([15232])
df4 = df4.reset_index()


"""
frame = df4[['Quan','TinhTrangBDS','DienTich','Phongngu','TenPhanKhu','SoTang','PhongTam','Loai','GiayTo','MaCanHo'
             ,'TinhTrangNoiThat' ,'HuongCuaChinh','HuongBanCong','DacDiem',"Gia"]]
            
"""
frame = df4[['DienTich','Phongngu','SoTang','PhongTam'
             ,'Gia']]

# Remove rows where 'DienTich' is equal to 'DienTich'
frame = frame[frame['DienTich'] != 'DienTich']

# Convert 'DienTich' column to float
frame['DienTich'] = frame['DienTich'].astype(float)

frame.Phongngu = frame.Phongngu.astype('float') 

frame.PhongTam = frame.PhongTam.astype('float') 
frame.Gia = frame.Gia.astype('float') 
frame.SoTang = frame.SoTang.astype('float') 

frame.describe()

frame['USD'] = round(frame['Gia']/24000,0)
rows  = frame[frame.DienTich > 500]
frame = frame.drop(index = rows.index)
rows  = frame[frame.SoTang > 81]
frame = frame.drop(index = rows.index)
frame['log_price'] = np.log(frame.USD)

print("The number of row after cleaning data:",len(frame))
frame.to_csv('dataset.csv',encoding="utf-8-sig",index=False)


# truc quan du lieu

def scatter(x,fig):
    plt.subplot(5,2,fig)
    plt.scatter(frame[x],frame['log_price'])
    plt.title(x+' vs logPrice')
    plt.ylabel('logPrice')
    plt.xlabel(x)


plt.figure(figsize=(10,12))

scatter('DienTich', 1)
scatter('Phongngu', 2)
scatter('SoTang', 3)
scatter('PhongTam', 4)

plt.tight_layout()

#plt.show()


bins = [42,67917,106250,40833333]



frame1 = df4[['Quan','TinhTrangBDS','DienTich','Phongngu','TenPhanKhu','SoTang','PhongTam','Loai','GiayTo','MaCanHo'
             ,'TinhTrangNoiThat' ,'HuongCuaChinh','HuongBanCong','DacDiem',"Gia"]]

frame1 = frame1[frame1['DienTich'] != 'DienTich']

# Replace 'DienTich' with the correct column name
frame1['DienTich'] = frame1['DienTich'].astype(float)
frame1.Phongngu = frame1.Phongngu.astype('float') 
frame1.PhongTam = frame1.PhongTam.astype('float') 
frame1.Gia = frame1.Gia.astype('float') 
frame1.SoTang = frame1.SoTang.astype('float') 
frame1['USD'] = round(frame1['Gia']/24000,0)
rows  = frame1[frame1.DienTich > 500]
frame1 = frame1.drop(index = rows.index)
rows  = frame1[frame1.SoTang > 81]
frame1 = frame1.drop(index = rows.index)
frame1['log_price'] = np.log(frame1.USD)

def Analyst(col):
    temp = frame1.copy()
    table = temp.groupby([col])['USD'].mean()
    temp = temp.merge(table.reset_index(), how='left',on=col)
    cars_bin=['low','Medium','High']
    frame1['range'] = pd.cut(temp['USD_y'],bins,right=False,labels=cars_bin)
    plt.rcParams['figure.figsize'] = (18, 8)
    df = pd.DataFrame(frame1.groupby([col,'range'])['USD'].mean().unstack(fill_value=0))
    df.plot.bar()
    plt.title('house Range vs Price')
    plt.show()

Analyst('Quan')
Analyst('TinhTrangBDS')
Analyst('TinhTrangNoiThat')
Analyst('Loai')


test = frame1[frame1.USD <frame1.USD.quantile(0.8)]
plt.rcParams['figure.figsize'] = (12, 6)

print(test.HuongBanCong.value_counts())
sns.boxplot(x=test.HuongBanCong, y=test.USD, palette=("plasma"))
plt.title('Huong ban cong vs Gia')
plt.show()

print(test.HuongCuaChinh.value_counts())
sns.boxplot(x=test.HuongCuaChinh, y=test.USD, palette=("plasma"))
plt.title('Huong cua chinh vs Gia')
plt.show()

print(test.Loai.value_counts())
sns.boxplot(x=test.Loai, y=test.USD, palette=("plasma"))
plt.title('Loai Can ho vs Gia')
plt.show()

print(test.GiayTo.value_counts())
sns.boxplot(x=test.GiayTo, y=test.USD, palette=("plasma"))
plt.title('GiayTo vs Gia')
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('dataset.csv')
df.describe()

frame1['Rank'] = frame1.USD.copy()
df_1 = frame1[frame1['USD'] < 50000]
df_2 = frame1[(frame1['USD'] >= 50000) & (frame1['USD'] < 100000)]
df_3 = frame1[(frame1['USD'] >= 100000) & (frame1['USD'] < 150000)]
df_4 = frame1[frame1['USD'] >= 150000]
df_2.head(2)


nan_values = df.isna()
nan_columns = nan_values.any()
columns_with_nan = df.columns[nan_columns].tolist()
columns_with_nan

df.isna().sum()/df.shape[0]*100

#drop cac cot du lieu theo %

per = 0.5 # Chọn xóa những cột dữ liệu có trên 50% data là NaN
df_dropped = df.dropna(axis=1,thresh=int(df.shape[0]*per))
df_dropped_2 = df_dropped.dropna(how='any')
#df_dropped_2 = df_dropped.dropna(axis=0,thresh=int(df.shape[1]*0.5))
df_dropped_2

## Do giá tiền Việt quá lớn nên mình sẽ chuyển về dạng USD 
df_x = df_dropped_2.iloc[:, 1:9]
df_x

df_y = df_dropped_2.iloc[:, 10]
df_y

#lay dummy data

data_dummies = pd.get_dummies(df_x, drop_first=True)
data_dummies = data_dummies.astype(float)
cols = data_dummies.columns.values
data_preprocessed = data_dummies[cols]
data_preprocessed

#scale data ve dang chuan 
scaler = StandardScaler()
scaler.fit(data_preprocessed)
data_preprocessed = scaler.transform(data_preprocessed)
data_preprocessed

#split data 80% train, va 20% test
X,X_test,Y,Y_test = train_test_split(data_preprocessed,df_y,test_size=0.2,random_state=365)
reg = LinearRegression().fit(X, Y)
reg.predict(X_test)
Y_test

Y_pre = reg.predict(X_test)
perc = np.abs((Y_pre - Y_test)/Y_test)
perc = perc.values*100
np.array([Y_pre, Y_test, perc])

data = {'Gia du doan':Y_pre,
        'Gia thuc':Y_test,
       '% sai lech': perc}
A = pd.DataFrame(data)
B = A.sort_values(by=['% sai lech'])
B

B['% sai lech'].describe()


# tinh sai lech trung binh 
(sum((reg.predict(X_test) - Y_test)**2)/len(Y_test))**0.5