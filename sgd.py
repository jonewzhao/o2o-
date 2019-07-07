#---*--encoding:utf-8-*

import os, sys, pickle

import numpy as np
import pandas as pd
import math
from datetime import date

from sklearn.model_selection import KFold,train_test_split,StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #随机决策森林
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss,roc_auc_score,auc,roc_curve
from sklearn.preprocessing import MinMaxScaler
dfoff = pd.read_csv('data/ccf_offline_stage1_train.csv')

dfon = pd.read_csv('data/ccf_online_stage1_train.csv')
dftest = pd.read_csv('data/ccf_offline_stage1_test_revised.csv')

'''
dfoff = pd.read_csv('data/aa.csv')

dfon = pd.read_csv('data/aa.csv')
dftest = pd.read_csv('data/aa.csv')
'''

'''
print('有优惠卷，购买商品：%d' % dfoff[(np.isnan(dfon['Date_received'])) & (np.isnan(dfon['Date']))].shape[0])

print('有优惠卷，未购商品：%d' % dfoff[(dfoff['Date_received'] != 'NaN') & (dfoff['Date'] == np.nan)].shape[0])
print('无优惠卷，购买商品：%d' % dfoff[(dfoff['Date_received'] == 'NaN') & (dfoff['Date'] != np.nan)].shape[0])
print('无优惠卷，未购商品：%d' % dfoff[(dfoff['Date_received'] == 'NaN') & (dfoff['Date'] == np.nan)].shape[0])
'''
#print('Discount_rate 类型：\n',dfoff['Discount_rate'].unique())

'''
特征：
打折类型：getDiscountType（）
折扣率：convertRate（）
满多少：getDiscountMan（）
减多少：getDiscountJian()

'''
def getDiscountType(row):

    if type(row)==float and np.isnan(row):
        return 'null'
    elif ':' in str(row):
        return 1
    else:
        return 0

def convertRate(row):
    """Convert discount to rate"""
    if type(row)==float and np.isnan(row):
        return 1.0
    elif ':' in str(row):
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

def getDiscountMan(row):
   if ':' in str(row):
       rows = row.split(':')
       return int(rows[0])
   else:
       return 0

def getDiscountJian(row):
   if ':' in str(row):
       rows = row.split(':')
       return int(rows[1])
   else:
       return 0

def processData(df):
# convert discount_rate
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)

    #print(df['discount_rate'].unique())
    return df


dfoff = processData(dfoff)
dftest = processData(dftest)

#print(dfoff.head(5))

'''
特征：
领券日期
独热编码

'''

def getWeekday(row):
   if np.isnan(float(row)):
       return float(row)
   else:
       return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).isoweekday()

dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

#print(dfoff['weekday'].dtypes)
# weekday_type :  周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x: 1 if x in [6,7] else 0 )
dftest['weekday_type'] = dftest['weekday'].apply(lambda x: 1 if x in [6,7] else 0)
#print(dfoff['weekday_type'])
# change weekday to one-hot encoding
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
#print(weekdaycols)

tmpdf = pd.get_dummies(dfoff['weekday'],dummy_na=False)
#print(tmpdf)
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'],dummy_na=False)
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf


def label(row):
    if type(row['Date_received'])==float and np.isnan(row['Date_received']):
        return -1
    if float(row['Date'])!=np.nan:
        td = pd.to_datetime(row['Date'],format='%Y%m%d')-pd.to_datetime(row['Date_received'],format='%Y%m%d')
        if td<=pd.Timedelta(15,'D'):
            return 1
    return 0

dfoff['label'] = dfoff.apply(label,axis=1)
#print(dfoff['label'].value_counts())


#划分训练集和验证集
df = dfoff[dfoff['label']!=-1].copy()
train = df[(df['Date_received']<20160516)].copy()
valid = df[(df['Date_received']>=20160516) & (df['Date_received']<=20160615)].copy()

print('Train Set: \n', train['label'].value_counts())
print('Valid Set: \n', valid['label'].value_counts())

original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian', 'weekday', 'weekday_type'] + weekdaycols
# model1
predictors = original_feature
print(predictors)


def check_model(data, predictors):
    classifier = lambda: SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        max_iter=100,
        shuffle=True,
        n_jobs=1,
        class_weight=None)

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])

    parameters = {
        'en__alpha': [0.001, 0.01, 0.1],
        'en__l1_ratio': [0.001, 0.01, 0.1]
    }

    folder = StratifiedKFold(n_splits=3, shuffle=True)

    grid_search = GridSearchCV(
        model,
        parameters,
        cv=folder,
        n_jobs=-1,
        verbose=1)
    grid_search = grid_search.fit(data[predictors],
                                  data['label'])

    return grid_search


if not os.path.isfile('1_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('1_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('1_model.pkl', 'rb') as f:
        model = pickle.load(f)

y_valid_pred = model.predict_proba(valid[predictors])
valid1 = valid.copy()
valid1['pred_prob'] = y_valid_pred[:,1]
valid1.head(2)

vg = valid1.groupby(['Coupon_id'])
aucs =[]
for i in vg:
    tmpdf = i[1]
    if len(tmpdf['label'].unique())!=2:
        continue
    fpr,tpr,thresholds = roc_curve(tmpdf['label'],tmpdf['pred_prob'],pos_label=1)
    aucs.append(auc(fpr,tpr))
print(np.average(aucs))

print("----train-----")
model = SGDClassifier(#lambda:
    loss='log',
    penalty='elasticnet',
    fit_intercept=True,
    max_iter=100,
    shuffle=True,
    alpha = 0.01,
    l1_ratio = 0.01,
    n_jobs=1,
    class_weight=None
)
model.fit(train[original_feature], train['label'])


# test prediction for submission
y_test_pred = model.predict_proba(dftest[predictors])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['Probability'] = y_test_pred[:,1]
dftest1.to_csv('submit.csv', index=False, header=False)
dftest1.head(5)
