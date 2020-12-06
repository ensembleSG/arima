# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:12:25 2020
@ 
"""
 
#import os
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

import statsmodels.api as sm
from pmdarima import auto_arima
from sklearn.metrics import r2_score
plt.style.use('seaborn-whitegrid')


def get_code(df,name):    
     code = df.query("name=='{}'".format(name))['code'].to_string(index=False)    
     code = code.strip()
     code = code+".KS"
     return code


code_data = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13',header=0)[0]
code_data # 2380 * 9 columns

code_data = code_data[['회사명','종목코드']]
code_data = code_data.rename(columns={'회사명':'name','종목코드':'code'})
code_data.code = code_data.code.map('{:06d}'.format)

#stock_code = get_code(code_data,'LG이노텍')
stock_code = get_code(code_data,'삼성전자')

stock_code


stock_data = pdr.get_data_yahoo(stock_code, start="2018-01-01")

'''
High: 장 중 제일 높았던 주가(고가)
Low: 장 중 제일 낮았던 주가 (저가)
Open: 장 시작 때 주가 (시가)
Close: 장 닫을 때 주가(종가)
Volume: 주식 거래량
Adj Close: 주식의 분할, 배당,배분 등을 고려해 조정한 종가
'''
# 473 * 6


# 장 종가 데이터를 학습데이터셋으로 이용
stock_data = stock_data[['Close']]
fig = stock_data.plot()

# 학습데이터와 테스트 데이터 분리 train / validation
# 5% test dataset (validation data) 지정
from sklearn.model_selection import train_test_split
stock_data_train,stock_data_test = train_test_split(stock_data,test_size=0.05,shuffle=False)


fig, ax = plt.subplots(1,2,figsize=(10,5))
fig.suptitle('Raw Data')    
sm.graphics.tsa.plot_acf(stock_data.values.squeeze(),lags=40,ax=ax[0])
sm.graphics.tsa.plot_pacf(stock_data.values.squeeze(),lags=40,ax=ax[1]) # modify not to generate graph twrice
    

# Differencing
#차분값 트레이닝 데이터 준비
# t - (t-1)
# 결측지 제거(missing value)
# 안정적인 시계열이란 시간의 추이와 관계없이 평균 및 분산이 불변하거나 시점 간의 공분산이 
#기준시점과 무관한 형태의 시계열이다
diff_stock_data_train = stock_data_train.copy()
diff_stock_data_train = diff_stock_data_train['Close'].diff()
diff_stock_data_train = diff_stock_data_train.dropna()
print('##### Raw Data #####')
print(stock_data_train)
print('#### Differenced data ######')
print(diff_stock_data_train)


# Differenced data plot
plt.figure(figsize=(12,8))
plt.subplot(211)
plt.plot(stock_data_train['Close'])
plt.legend(['Raw Data (Nonstationary'])
plt.subplot(212)
plt.plot(diff_stock_data_train,'orange') # first difference ( t - t(1-1))
plt.legend(['Differnced Data (Stationary'])
plt.show()

#ACF , PACF plot
fig,ax = plt.subplots(1,2,figsize=(10,5))
fig.suptitle('Differenced Data')
sm.graphics.tsa.plot_acf(diff_stock_data_train.values.squeeze(),lags=40,ax=ax[0])
sm.graphics.tsa.plot_pacf(diff_stock_data_train.values.squeeze(),lags=40,ax=ax[1]) # modify not to generatoe graph write


# Parameter search (ESstimate Parameters)
# Auto Diagnosis Check - ARIMA
# ARIMA model fitting
# The (p,d,q) order of the model for the number of AR parameters,
# diffrences, and MA parameters to use
# AIC ?
auto_arima_model = auto_arima(stock_data_train,start_p =1, start_q=1,
                              max_p=3,max_q=3,seasonal=False,
                              d=1,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=False
                              )

summary = auto_arima_model.summary()

prediction = auto_arima_model.predict(len(stock_data_test),return_conf_int=True)

predicted_value = prediction[0]
predicted_ub = prediction[1][:,0]
predicted_lb = prediction[1][:,1]
predict_index = list(stock_data_test.index)
r2 = r2_score(stock_data_test,predicted_value)

print(predict_index[0])
#print(predict_index)


fig, ax = plt.subplots(figsize=(12,6))
stock_data.plot(ax=ax)
ax.vlines('2020-11-30',50000,75000,linestyle='--',color='r',label='Start of Forecast');
ax.plot(predict_index,predicted_value,label ='Prediction')
ax.fill_between(predict_index,predicted_lb,predicted_ub,color = 'k' , alpha = 0.1, label ='0.95 Prediction Interval ')
ax.legend(loc='upper left')
plt.suptitle(f'ARIMA {auto_arima_model.order},Prediction Results (r2_score: {round(r2,2)} )')
plt.show()