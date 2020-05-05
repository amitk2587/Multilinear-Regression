import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
Startups = pd.read_csv("D:/Data_Science/Data_Sci_Assignment/Multi Linear Regression/50_Startups.csv")
Startups.columns

# Assigning category code to state
Startups['State'] = Startups['State'].astype("category").cat.codes


#Check the correlation the correlation 
Startups.corr()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Startups)

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols("Startups['Profit']~Startups['Administration']+Startups['Marketing Spend']+Startups['R&D Spend']",data=Startups).fit() 
ml1.summary()

#intercept are in significant
np.mean(ml1.resid) #0
np.sqrt(sum(ml1.resid**2)/49) #8945.24

#to check for influential record
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

#removing the influential records
Startups = Startups.drop(Startups.index[45],axis=0)

# Preparing model  
x = smf.ols("Startups['Administration']~Startups['Profit']+Startups['Marketing Spend']+Startups['R&D Spend']",data=Startups).fit().rsquared               
vif_adm =1/(1-x)
vif_adm

y = smf.ols("Startups['Marketing Spend']~Startups['Profit']+Startups['Administration']+Startups['R&D Spend']",data=Startups).fit().rsquared
vif_Mark_Spend = 1/(1-y)
vif_Mark_Spend

#Startups['Administration'] has more insignificant value for intercept
#vif for vif_Mark_Spend is more

# Added varible plot 
sm.graphics.plot_partregress_grid(ml1)
#AVP of Administration is insignificant 


#new model 
new_ml1 = smf.ols("Startups['Profit']~Startups['Marketing Spend']+Startups['R&D Spend']",data=Startups).fit() 
new_ml1.summary()

#influential plot
sm.graphics.influence_plot(new_ml1)
Startups = Startups.drop(Startups.index[[45,49]],axis=0)

#new model after removing influential records
new_ml1 = smf.ols("Startups['Profit']~Startups['Marketing Spend']+Startups['R&D Spend']",data=Startups).fit() 
new_ml1.summary()


#intercept are in significant
np.mean(new_ml1.resid) #0
np.sqrt(sum(new_ml1.resid**2)/49) #8601.16

Profit_pred = new_ml1.predict(Startups)

#AVP for final model
sm.graphics.plot_partregress_grid(new_ml1)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(Startups['Profit'],Profit_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(Startups['Profit'],new_ml1.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(new_ml1.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(new_ml1.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(Profit_pred,new_ml1.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
Startups_train,Startups_test  = train_test_split(Startups,test_size = 0.2) # 20% size


#model for traning
Startups_train_model = smf.ols("Startups_train['Profit'] ~ Startups_train['Marketing Spend']+Startups_train['R&D Spend']",data=Startups_train).fit() 
Startups_train_model.summary()


np.mean(Startups_train_model.resid) # 0
np.sqrt(sum(Startups_train_model.resid**2)/49) # 7978


#model for testing
Startups_test_model = smf.ols("Startups_test['Profit'] ~ Startups_test['Marketing Spend']+Startups_test['R&D Spend']",data=Startups_test).fit() 
Startups_test_model.summary()


np.mean(Startups_test_model.resid) # 0
np.sqrt(sum(Startups_test_model.resid**2)/49) # 6918