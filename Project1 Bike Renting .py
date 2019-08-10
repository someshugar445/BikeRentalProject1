
# coding: utf-8

# In[1]:


# Import all the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from sklearn.metrics import accuracy_score
import os
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime


# In[2]:


# Get current directory
dirpath = os.getcwd()
print("current directory is : " + dirpath)


# In[3]:


# Get file path
day_file = "/home/someshugar/Project 1/day.csv"


# In[4]:


# Read file  using pandas and check shape
df = pd.read_csv(day_file)
print(df.shape)


# In[5]:


# View and analyse head of the Dataframe
df.head()


# In[6]:


# Drop instant column
df = df.drop(['instant'], axis=1)


# In[7]:


df.dtypes


# In[8]:


# Date time conversion
df.dteday = pd.to_datetime(df.dteday, format='%Y-%m-%d')


# In[9]:


df['day'] = df['dteday'].dt.day


# In[10]:


# Categorical variables
for column in ['season', 'holiday', 'weekday', 'workingday', 'weathersit','yr']:
    df[column] = df[column].astype('category')


# In[11]:


df.dtypes


# 
# Also, we’ll already delete the fields dteday, instant and yr. We won’t need instant since it’s only an index.
# yr is not important because years don’t ‘repeat’ like months or days of the week, so it doesn’t
# really add a prediction element to future information.

# In[12]:


df.describe()


# In[13]:


# Find columns with NaN
df.isnull().sum(axis=0)


# In[14]:


df.info()


# # Exploratory Data Analysis
# 
# Bike sharing utilization over the two years
# The objective of this Case is to Predication of bike rental count on daily based on the environmental and seasonal settings.

# In[15]:


# Total_bikes rented count per day
fig, ax = plt.subplots(figsize=(15,5))
fig = sns.barplot(x = df.dteday, y = df.cnt,color = 'steelblue')    .axes.set_xticklabels(['March-2011', 'May-2011', 'June-2011', 'Aug-2011', 'Oct-2011', 'Dec-2011',
                          'Feb-2012', 'Apr-2012','Jun-2012','Aug-2012','Oct-2012','Dec-2012'])
ax.set(xlabel='Days', ylabel='Number of bikes rented', title='Rented bikes per day')
plt.xticks([60,120,180,240,300,360,420,480,540,600,660,720])
plt.savefig('Figure1.png')
plt.show()


# In[16]:


# Box plot of Rented bikes in different seasons
fig, ax = plt.subplots(figsize=(14,8))
fig = sns.boxplot(x='season', y='cnt', data=df, ax=ax)
ax.set(xlabel='Seasons', ylabel='Number of bikes rented', title='Rented bikes in different seasons')
seasons=['springer','summer', 'fall', 'winter']
ax.set_xticklabels(seasons)
plt.savefig('Figure2.png')
plt.show()


# In[17]:


fig, ax = plt.subplots(figsize=(14,8))
sns.boxplot(x='mnth', y='cnt',hue='workingday', data=df, ax=ax)
ax.set(xlabel='Month', ylabel='Number of bikes rented', title='Rented bikes in different months')
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
ax.set_xticklabels(months)
plt.savefig('Figure3.png')
plt.show()


# In[18]:


fig, ax = plt.subplots(figsize=(14,8))
sns.boxplot(x='mnth', y='cnt',hue='holiday', data=df, ax=ax)
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
ax.set(xlabel='Month', ylabel='Number of bikes rented', title='Rented bikes in different months hue=holiday')
ax.set_xticklabels(months)
plt.savefig('Figure4.png')
plt.show()


# In[19]:


fig, ax = plt.subplots(figsize=(14,8))
sns.boxplot(x='mnth', y='cnt',hue='weekday', data=df, ax=ax)
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
ax.set_xticklabels(months)
ax.set(xlabel='Month', ylabel='Number of bikes rented', title='Rented bikes in different months hue=weekday')
plt.savefig('Figure5.png')
plt.show()


# In[20]:


fig, ax = plt.subplots(figsize=(14,8))
sns.boxplot(x='weathersit', y='cnt', data=df, ax=ax)
ax.set(xlabel='weather', ylabel='Number of bikes rented', title='Rented bikes in different weathers')
weather = ['clear+fewclouds', 'cloudy', 'light rain','heavy rain']
ax.set_xticklabels(weather)
plt.savefig('Figure6.png')
plt.show()


# In[21]:


fig= plt.subplots(figsize=(14,8))
axes1 = plt.subplot(2, 2, 1)
axes2 = plt.subplot(2, 2, 2)
axes3 = plt.subplot(2, 2, 3)
axes4 = plt.subplot(2, 2, 4)

axes1.scatter(x='temp', y='cnt', data=df)
axes2.scatter(x='atemp', y='cnt', data=df)
axes3.scatter(x='hum', y='cnt', data=df)
axes4.scatter(x='windspeed', y='cnt', data=df)

axes1.set(xlabel='Normalized temp in Celsius', ylabel='Number of bikes rented', title='Rented bikes in different temperatures')
axes2.set(xlabel=' Normalized feeling temp in Celsius', ylabel='Number of bikes rented', title='Rented bikes in different feeling temperatures')
axes3.set(xlabel='Normalized humidity', ylabel='Number of bikes rented', title='Rented bikes in different humidity')
axes4.set(xlabel=' Normalized wind speed', ylabel='Number of bikes rented', title='Rented bikes in different windspeed')
plt.tight_layout()
plt.savefig('Figure7.png')
plt.show()


# # Correlation Analysis
# A correlation analysis will allow to identify relationships between the dataset variables. A plot of their distributions
# highlighting the value of the target variable might also reveal some patterns.

# In[22]:


df_columns =['mnth', 'casual', 'registered','dteday']
df = df.copy()
days_df_corr = df.drop(df_columns, axis=1)
for column in days_df_corr.columns:
    days_df_corr[column] = days_df_corr[column].astype('float')
    
plt.figure(figsize=(12, 10))
sns.heatmap(days_df_corr.corr(), 
            cmap=sns.diverging_palette(220, 20, n=7), vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)
plt.savefig('Figure8.png')
plt.show()


# From the above correlation plots actual temperature is more correlated with bike rentals count, humidity and wind speed 
# are also slightly correlated.Temperature and actual are highly  correlated.
# 
# Another problem are our registered and casual variables. Think about it: imagine that you are trying to predict the total number of users for tomorrow. While you can have data such as the month, the weekday, the temperature and the weather condition, it’s impossible to have the number of registered and casual users because this is exactly what you are trying to predict.
# 
# Also, since the count is a decomposition of these two variables, we could have problems if they remain on the data set. So, let’s get rid of them too.
# 

# In[23]:


df = df.drop(['dteday', 'atemp', 'casual', 'registered'], axis=1)


# In[24]:


df.head()


# # One Hot Encoding
# Since we have categorical values in our data set, we need to ‘tell’ our algorithm that classes have equal weight for our analysis. For instance: our weekdays are represented by numbers from 0 to 6. But we can’t really say that a 6 is better than a 5 here.
# 
# A way to change this perspective is using the one hot encoding technique. This is a process by which we convert categorical variables into binary categories. By the way, when we apply one hot encoding, it’s important to left one variable out to avoid multicollinearity. 
# 
# 

# In[25]:


df_dummy = df

def dummify_dataset(df, column):       
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column, drop_first=True)],axis=1)
    df = df.drop([column], axis=1)
    return df

columns_to_dummify = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
for column in columns_to_dummify:
    df_dummy = dummify_dataset(df_dummy, column)
    
df_dummy.head()


# In[26]:


y = df_dummy['cnt']
X = df_dummy.drop(['cnt'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)


# In[27]:


model = ExtraTreesRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



# In[28]:


# Plot the residuals

def rmsle(y_test, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test))**2))


residuals = y_test-y_pred
fig, ax = plt.subplots()
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text('Residual Plot | Root Squared Mean Log Error: ' + str(rmsle(y_test, y_pred)))
plt.savefig('Figure9.png')
plt.show()


# In[29]:


print("RMSLE: ", rmsle(y_test, y_pred))


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
my_model = DecisionTreeRegressor()
my_model.fit(X_train, y_train)
y_pred = my_model.predict(X_test)   


# In[31]:


print (mean_absolute_error(y_test,y_pred)) 
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))


# In[32]:


print("RMSLE: ", rmsle(y_test,y_pred))


# In[34]:


residuals = y_test-y_pred
fig, ax = plt.subplots()
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text('Residual Plot | Root Squared Mean Log Error: ' + str(rmsle(y_test,y_pred)))
plt.savefig('Figure10.png')
plt.show()

