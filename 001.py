#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from datetime import datetime, timedelta
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# In[47]:


pip install xgboost


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


pip install fastparquet


# In[3]:


import fastparquet


# In[27]:


# import zipfile
# import pprint

# # Specify the path to the zip file and the target directory
# zip_file_path = 'C:/Users/HP/Downloads/Structured_Data_Assignment .zip'
# extracted_dir = 'C:/Users/HP/Downloads/Extracted_Data'

# # List of file names to extract from the zip archive
# files_to_extract = ['Structured_Data_Assignment /train.parquet','Structured_Data_Assignment /test.parquet']


# # with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
# #     pprint.pprint(zip_ref.namelist())

# # Open the zip file in read mode
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     # Extract specific files to the target directory
#     for file_to_extract in files_to_extract:
#         zip_ref.extract(file_to_extract, extracted_dir)


# In[4]:


os.chdir('C:/Users/HP/Downloads/Structured_Data_Assignment')


# In[5]:


data=pd.read_parquet('train.parquet', engine='fastparquet')
datacopy=data.copy()


# In[6]:


data


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.head(25)


# In[10]:


data.describe()


# In[11]:


data.isnull().sum()


# In[12]:


data.duplicated().sum()


# In[13]:


data=data.drop_duplicates()


# In[14]:


data.duplicated().sum()


# In[15]:


data.dtypes


# In[16]:


data['Incident'].unique()


# In[17]:


data.Incident.value_counts()


# In[18]:


data.Date.value_counts()


# In[19]:


data_positive=data[data['Incident']=='TARGET DRUG']


# In[20]:


data_positive.head()


# In[21]:


data_positive.shape


# In[22]:


negative=data[~(data['Patient-Uid'].isin(data_positive['Patient-Uid']))]
data_negative=negative.groupby('Patient-Uid').tail(1)
data_negative


# In[23]:


data_negative.shape


# In[24]:


data_positive['Prescription count']=data_positive.groupby('Patient-Uid')['Date'].cumcount()
data_negative['Prescription count']=data_negative.groupby('Patient-Uid')['Date'].cumcount()
data_positive.tail()


# In[25]:


data_negative.tail()


# In[26]:


#find the difference between recent prescription date and prediction date
pred_date=pd.to_datetime('today')+pd.DateOffset(days=30)
data_positive['timediff']=(pred_date-data_positive.groupby('Patient-Uid')['Date'].transform('max')).dt.days
data_negative['timediff']=(pred_date-data_negative.groupby('Patient-Uid')['Date'].transform('max')).dt.days


# In[27]:


data_positive.head()


# In[28]:


data_negative.head()


# In[29]:


#make new dataframe by concatinating data_positive and data_negative
df_new=pd.concat([data_positive,data_negative])
df_new.head()


# In[30]:


df_new.shape


# In[31]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_new[['Prescription count', 'timediff']], df_new['Incident'] == 'TARGET DRUG', test_size = 0.25, random_state=42)

     


# In[32]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[33]:


#model building to train the data
xgb_classifier =  XGBClassifier(random_state=42)
xgb_classifier.fit(X_train,y_train)


# In[34]:


#predict the test data
y_pred=xgb_classifier.predict(X_test)


# In[35]:


#now evaluate the model with confusion matrix
from sklearn.metrics import confusion_matrix


# In[36]:


conf_matrix=confusion_matrix(y_test,y_pred)
conf_matrix


# In[37]:


plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix,annot=True,fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[38]:


#evaluate the model with classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[62]:


#finding F1 score
F1_score=f1_score(y_test,y_pred)
F1_score


# In[40]:


# evaluating model by roc_auc curve
from sklearn.metrics import roc_curve, auc
fpr,tpr, thresold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr, label = 'AUC = %0.3f' % roc_auc)
plt.plot([0,1],[0,1],'--')
plt.title('ROC_AUC curve')
plt.legend(loc='lower right')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()


# In[ ]:





# In[41]:


#Load test data
test_data=pd.read_parquet('test.parquet', engine='fastparquet')
test_data_copy=test_data.copy()


# In[42]:


test_data.shape


# In[43]:


test_data.head()


# In[44]:


test_data.tail()


# In[45]:


test_data.columns


# In[46]:


test_data.describe()


# In[47]:


test_data.isnull().sum()


# In[48]:


test_data.duplicated().sum()


# In[49]:


test_data=test_data.drop_duplicates()


# In[50]:


test_data.duplicated().sum()


# In[51]:


#check the datatypes
test_data.dtypes


# In[52]:


#find the unique values in the 'Incident' column
test_data['Incident'].unique()


# In[53]:


#find the counts of each values in 'Incident' column
test_data['Incident'].value_counts()


# In[54]:


positive_testdata=test_data[test_data['Incident']=="TARGET DRUG"]
positive_testdata.head()


# In[55]:


positive_testdata.shape


# In[75]:


negative_=test_data[~(test_data['Patient-Uid'].isin(positive_testdata['Patient-Uid']))]
negative_testdata=negative_.groupby('Patient-Uid').tail(1)
negative_testdata


# In[57]:


negative_testdata.shape


# In[58]:


#get the count of previous prescriptions within specific time intervals
positive_testdata['Prescription count']=positive_testdata.groupby('Patient-Uid')['Date'].cumcount()
negative_testdata['Prescription count']=negative_testdata.groupby('Patient-Uid')['Date'].cumcount()
positive_testdata.head()


# In[59]:


negative_testdata.tail()


# In[60]:


#get the difference between the most recent prescription and the prediction date
pred_date=pd.to_datetime('today')+pd.DateOffset(days=30)
positive_testdata['timediff']=(pred_date-positive_testdata.groupby('Patient-Uid')['Date'].transform('max')).dt.days
negative_testdata['timediff']=(pred_date-negative_testdata.groupby('Patient-Uid')['Date'].transform('max')).dt.days


# In[61]:


positive_testdata.head()


# In[62]:


negative_testdata.head()


# In[63]:


#create new dataset by concating positive and negative sets
dfnew=pd.concat([positive_testdata,negative_testdata])
dfnew.head()


# In[64]:


dfnew.shape


# In[65]:


data.drop_duplicates(inplace=True)


# In[66]:


test_data['Prescription count']=test_data.groupby('Patient-Uid')['Date'].cumcount()
test_data['timediff']=(pred_date-test_data.groupby('Patient-Uid')['Date'].transform(max)).dt.days


# In[76]:


test_data_prediction = xgb_classifier.predict(dfnew[['Prescription count', 'timediff']])
test_data_prediction


# In[78]:


Final2_sub = pd.DataFrame({'Patient-Uid': dfnew['Patient-Uid'], 'Prediction': test_data_prediction})
Final2_sub.head()


# In[79]:


Final2_sub.shape


# In[80]:


Final2_sub.to_csv('Final2_submission.csv', index = False)


# In[67]:





# In[68]:





# In[69]:





# In[70]:





# In[71]:





# In[ ]:





# In[72]:





# In[ ]:




