import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

computer = pd.read_csv("D:/Data_Science/Data_Sci_Assignment/Multi Linear Regression/Computer_Data.csv")

computer= computer.drop(["Unnamed: 0"],axis=1)

computer.speed.value_counts()
computer.hd.value_counts()
computer.ram.value_counts()
computer.screen.value_counts()
computer.cd.value_counts()
computer.multi.value_counts()
computer.premium.value_counts()
computer.ads.value_counts()
computer.trend.value_counts()
## in computer data set ram, cd, screen, multi, premium are categorical data. So converting them into factor

import seaborn as sn
sn.pairplot(computer)

corr_values = computer.corr()


##The correlation between ads--trend is greater than price--ads and price--trend.
## But the scatter plot shows that it has somewhat curvilinear shape 

## The same way, speed-- trend has higher correlation value than price--speed and price --trend
##But the scatter plot doesnt show any linearity problem between speed and trend 

## The correlation betweeen hd--ads is higher than price--hd and lower than price--ads. 
## The scatter plot shows kind off linearity with high scatter of data.

## The correlation between hd--trend is higher than price--hd and also price--trend.
##The scatter plot shows kind off linearity, with high scatter of data.

## Knowing this let us build the model

computer_ram= pd.get_dummies(computer["ram"])
computer_cd=pd.get_dummies(computer["cd"])
computer_screen = pd.get_dummies(computer["screen"])
computer_premium = pd.get_dummies(computer["premium"])
computer_multi = pd.get_dummies(computer["multi"])

computer_ram.columns=["two","four","eight","sixteen","twentyfour","thirtytwo"]
computer_screen.columns=["fourteen","fifteen","seventeen"]
computer_multi.columns = ["no_multi","yes_multi"]
computer_cd.columns=["no_cd","yes_cd"]
computer_premium.columns=["no_pre","yes_pre"]

computer_final= pd.concat([computer,computer_cd,computer_ram,computer_multi,computer_premium,computer_screen],axis=1)

computer_final=computer_final.drop(["ram"],axis=1)
computer_final=computer_final.drop(["cd"],axis=1)
computer_final=computer_final.drop(["screen"],axis=1)
computer_final= computer_final.drop(["multi"],axis=1)
computer_final= computer_final.drop(["premium"],axis=1)

## splitting data into test data and train data
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(computer_final)
import statsmodels.formula.api as smf

#model1
m1= smf.ols("price~speed+hd+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+ads+trend", data= train_data).fit()
m1.summary() ## 0.792(r squared)

## all the variables are significant. So, there is no collinearity problem.


##transforming the model m1 for better r squared value

##transform 1
## applying sqrt to speed
m1_fin= smf.ols("price~np.sqrt(speed)+hd+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+ads+trend", data= train_data).fit()
m1_fin.summary()## 0.798(r squared)

##transform2
## applying sqrt to speed, hd and log to price
m2_fin = smf.ols("np.log(price)~np.sqrt(speed)+hd+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+ads+trend",data= train_data).fit()
m2_fin.summary() ## 0.811( r squared)

## transform3
## applying sqrt to speed,hd and log to price
m3_fin = smf.ols("np.log(price)~np.sqrt(speed)+np.sqrt(hd)+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+ads+trend",data = train_data).fit()
m3_fin.summary() ##0.819 (r squared)

## transform4
## Applying quadratic to speed and hd, sqrt to speed,hd,ads and log to output variable price
m4_fin = smf.ols("np.log(price)~np.sqrt(speed)+speed*speed+np.sqrt(hd)+(hd*hd)+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+np.sqrt(ads)+trend",data = train_data).fit()
m4_fin.summary() ##0.827 (r squared)

##tranform5
## Applying quadratic to speed,hd. Log to hd, sqrt to speed,ads and log to output variable.
m5_fin = smf.ols("np.log(price)~np.sqrt(speed)+(speed*speed)+np.log(hd)+(hd*hd)+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+np.sqrt(ads)+trend",data = train_data).fit()
m5_fin.summary() ## 0.830 ( r squared)

## taking transform1 and predicting the values and train,test rmse
m1_fin= smf.ols("price~np.sqrt(speed)+hd+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+ads+trend", data= train_data).fit()
m1_fin.summary()## 0.798

##train prediction
pred = m1_fin.predict(train_data)

##train residuals
m1_finres = train_data["price"]-pred

##train rmse
m1_finrmse = np.sqrt(np.mean(m1_finres*m1_finres)) ## 261

##test prediction
test_pre = m1_fin.predict(test_data)

## test residuals
m1_fintestres = test_data["price"]-test_pre

##test rmse
m1_fintestrmse= np.sqrt(np.mean(m1_fintestres*m1_fintestres))## 271 (4)

## taking transform2 and predicting the values and train, test rmse
m2_fin = smf.ols("np.log(price)~np.sqrt(speed)+hd+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+ads+trend",data= train_data).fit()
m2_fin.summary() ##0.811

##train prediction
pred2= m2_fin.predict(train_data)
pred21 = np.exp(pred2)

##train residuals
m2_finres = train_data["price"]- pred21

##train rmse
m2_finrmse = np.sqrt(np.mean(m2_finres*m2_finres)) ##253

## test prediction
m2_fintestpred1 = m2_fin.predict(test_data)
m2_fintestpred11= np.exp(m2_fintestpred1)
##test residuals
m2_fintestres = test_data["price"]- m2_fintestpred11
## test rmse
m2_fintestrmse = np.sqrt(np.mean(m2_fintestres*m2_fintestres))## 248 (5)


##taking transform3 and predicting the values and train, test rmse
m3_fin = smf.ols("np.log(price)~np.sqrt(speed)+np.sqrt(hd)+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+ads+trend",data = train_data).fit()
m3_fin.summary() ##0.814

##train prediction
m3_finpred = m3_fin.predict(train_data)
m3_finpred1 = np.exp(m3_finpred)

##train residuals
m3_finres = train_data["price"]-m3_finpred1

##train rmse
m3_finrmse = np.sqrt(np.mean(m3_finres*m3_finres)) ##261

## test prediction
m3_fintestpred = m3_fin.predict(test_data)
m3_fintestpred1 = np.exp(m3_fintestpred)

##test residuals
m3_fintestres=  test_data["price"]-m3_fintestpred1

##test rmse
m3_fintestrmse = np.sqrt(np.mean(m3_fintestres*m3_fintestres)) ##241 (4)

##taking transform4 and predicting the values and train, test rmse
m4_fin = smf.ols("np.log(price)~np.sqrt(speed)+speed*speed+np.sqrt(hd)+(hd*hd)+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+np.sqrt(ads)+trend",data = train_data).fit()
m4_fin.summary() ##0.827 (r squared)

##train prediction 
m4_finpred = m4_fin.predict(train_data)
m4_finpred1 = np.exp(m4_finpred)

##train residuals
m4_finres= train_data["price"]-m4_finpred1 

## train rmse
m4_finrmse = np.sqrt(np.mean(m4_finres*m4_finres)) ## 235

##test prediction
m4_finptestpred = m4_fin.predict(test_data)
m4_fintestpred1= np.exp(m4_finptestpred)
## test residuals
m4_fintestres = test_data["price"]- m4_fintestpred1
##test rmse
m4_fintestrmse = np.sqrt(np.mean(m4_fintestres*m4_fintestres)) ## 233 (7)


## taking transform5 and predicting the values and train, test rmse

m5_fin = smf.ols("np.log(price)~np.sqrt(speed)+(speed*speed)+np.log(hd)+(hd*hd)+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+np.sqrt(ads)+trend",data = train_data).fit()
m5_fin.summary() ## 0.823(r squared)

## train predicition
m5_finpred = m5_fin.predict(train_data)
m5_finpred1 = np.exp(m5_finpred)

##train residuals
m5_finres = train_data["price"]- m5_finpred1

##train rmse
m5_finrmse = np.sqrt(np.mean(m5_finres*m5_finres))## 238

## test prediction
m5_fintestpred = m5_fin.predict(test_data)
m5_fintestpred1 = np.exp(m5_fintestpred)
## test residuals
m5_finres = test_data["price"]-m5_fintestpred1
## test rmse
m5_finrmse = np.sqrt(np.mean(m5_finres*m5_finres)) ##231 (7)


##Looking at all the r square and rmse values we decide the final model.
## The selected model has r square of 0.814 and train rmse is 245 and test rmse is 241. So, the final model is:

fin = smf.ols("np.log(price)~np.sqrt(speed)+np.sqrt(hd)+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+ads+trend",data = train_data).fit()
fin.summary() ##0.814

## Training the model with the whole data
final = smf.ols("np.log(price)~np.sqrt(speed)+np.sqrt(hd)+two+four+eight+sixteen+twentyfour+thirtytwo+fourteen+fifteen+seventeen+no_cd+yes_cd+no_multi+yes_multi+no_pre+yes_pre+ads+trend",data = computer_final).fit()
final.summary() ## 0.814 ( r square)

final_pred= final.predict(computer_final)

##validating
###Linearity
plt.scatter(computer_final["price"],final_pred,c='r');plt.xlabel("Actual values"); plt.ylabel("Fitted values")
## it is linear

## Homoscadasticity
plt.scatter(final_pred,final.resid_pearson,c='r'),plt.axhline(y=0,color="blue");plt.xlabel("Fitted values");plt.ylabel("Residuals")
## Little Homoscadasticity

##Normality
## Plotting the historgram to see if the errors are normally distributed or not

plt.hist(final.resid_pearson)
## Because of the outliers present in data,it is showing the errors are not mormally distributed

##Plotting QQplot to properly visualise if the errors are normally distributed or not
import pylab
import scipy.stats as sc

sc.probplot(final.resid_pearson, dist="norm", plot=pylab)
## the data is normally distributed but there are outliers so the ends of the graphs are not on the same line


