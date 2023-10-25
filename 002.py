#!/usr/bin/env python
# coding: utf-8

# ### Problem 2 - 
# Drugs are generally administered/prescribed by the physicians for a certain
# period of time or they are administered at regular intervals, but for various reasons patients
# might stop taking the treatment . Consider following example for better understanding
# Let’s say you get a throat infection, the physician prescribes you an antibiotic for 10 days,
# but you stop taking the treatment after 3 days because of some adverse events.
# In the above example ideal treatment duration is 10 days but patients stopped taking
# treatment after 3 days due to adverse events. Patients stopping a treatment is called dropoff.
# We want to study dropoff for “Target Drug”, the aim is to generate insights on what events
# lead to patients stopping on “Target Drug”.
# Assume ideal treatment duration for “Target Drug” is 1 year, come up with analysis showing
# how drop-off rate is, dropoff rate is defined as number of patients dropping off each month.
# Then come up with analysis to generate insights on what events are driving a patient to stop
# taking “Target Drug”.

# In[3]:


#import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[4]:


os.chdir('C:/Users/HP/Downloads/Structured_Data_Assignment')


# In[6]:


train_data=pd.read_parquet('train.parquet', engine='fastparquet')
datacopy=train_data.copy()


# In[7]:


train_data.head()


# In[9]:


#finding who all are taking 'Target drug'
targetdata=train_data[train_data['Incident']=='TARGET DRUG']
targetdata.head()


# In[10]:


targetdata.shape


# In[11]:


# Dropoff rate by month is calculated
targetdata['Date']=pd.to_datetime(targetdata['Date'])
targetdata['Month']=targetdata['Date'].dt.month
dropoffrate=targetdata.groupby('Month')['Patient-Uid'].nunique().diff().fillna(0)


# In[12]:


#visualization of dropoff rate
plt.figure(figsize=(10,6))
dropoffrate.plot(kind='bar')
plt.xlabel='Month'
plt.ylabel='Drop_off_count'
plt.title='Target Drug-Drop off rate'
plt.show()


# In[15]:


#Analyze events leading to dropp-off
drop_reasons=train_data[train_data['Patient-Uid'].isin(targetdata['Patient-Uid'])]
drop_reasons=drop_reasons[drop_reasons['Date']<drop_reasons.groupby('Patient-Uid')['Date'].transform('max')]
drop_reasons=drop_reasons[drop_reasons['Incident']!='TARGET DRUG']


# In[17]:


#calculating the frequency of each event leading to drop-off
freq_of_event = drop_reasons['Incident'].value_counts()


# In[19]:


# Plotting the events leading to drop-off

plt.figure(figsize=(10,6))
freq_of_event.plot(kind='bar')
plt.xlabel='Event'
plt.ylabel='Frequency'
plt.title='Events that lead to drop off of Target Drug'
plt.show()


# In[ ]:




