#!/usr/bin/env python
# coding: utf-8

# # TASK:5  Credit card fraud detection

# In[2]:


# import necessary dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
credit_card_data = pd.read_csv("creditcard.csv")
credit_card_data


# # Data analysis

# In[3]:


ccd = credit_card_data
ccd.info()


# In[4]:


# no object data type so we wouldn't need any conversions in here
ccd.isnull().sum()


# In[5]:


# no null values, that's great
# distribution of fraudulent and legitimate classes
ccd['Class'].value_counts()


# # Data Visualization
# 

# In[6]:


# visualizing the class distribution in percentage
print((ccd.groupby('Class')['Class'].count()/ccd['Class'].count())*100)
((ccd.groupby('Class')['Class'].count()/ccd['Class'].count())*100).plot.pie()


# In[7]:


classes = ccd['Class'].value_counts()
normal_value = round(classes[0]/ccd['Class'].count()*100,2)
fraud_values = round(classes[1]/ccd['Class'].count()*100,2)
print(normal_value)
print(fraud_values)


# In[8]:


# let's check tthe correlation of the features
corr = ccd.corr()
corr


# In[9]:


# plotting the heatmap for the correlation
plt.figure(figsize=(27,19))
sns.heatmap(corr, cmap = 'spring', annot= True )
plt.show()


# In[10]:


# separte the data according to type of transaction i.e. fraud or legit
legit = ccd[ccd.Class == 0]
fraud = ccd[ccd.Class==1]
legit.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


# we can observe that the mean amount spent for fraud transactions is actually more than for the legit ones
ccd.groupby('Class').describe()


# In[13]:


ccd.groupby('Class').mean()


# In[14]:


# there's a significant difference in the mean value for our normal transaction and mean value for our fraud transactions
# now to balance the data for legit and fraud transaction value points 
# we will use sampling for creating a new dataset of normal transactions with 492 entries being selected randomly out of 284315
normal_sample = legit.sample(n=492)
# now merge the two datasets for fraud and legit transactions with equal number of sampl points
new_dataset = pd.concat([normal_sample, fraud], axis = 0) # axis =0 species row wise joining of the datasets l
new_dataset


# In[15]:


new_dataset['Class'].value_counts()


# In[16]:


new_dataset.groupby('Class').mean() 


# # separating the features and target variables

# In[21]:


x = new_dataset.drop('Class', axis=1)
y = new_dataset['Class']
x.shape


# In[22]:


y.shape


# # splitting the data into training and testing data

# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 3, stratify = y)
#accumulating all the column names under one variable
cols = list(x.columns.values)


# In[24]:


normal_entries = new_dataset.Class==0
fraud_entries = new_dataset.Class==1

plt.figure(figsize=(20,70))
for n, col in enumerate(cols):
    plt.subplot(10,3,n+1)
    sns.histplot(x[col][normal_entries], color='blue', kde = True, stat = 'density')
    sns.histplot(x[col][fraud_entries], color='red', kde = True, stat = 'density')
    plt.title(col, fontsize=17)
plt.show()


# In[25]:


model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_train)
pred_test = model.predict(x_test)


# # Model evaluation

# In[26]:


# creating confusion matrix
from sklearn.metrics import confusion_matrix
def Plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test,pred_test)
    plt.clf()
    plt.show()


# In[27]:


# accuracy on training data
acc_score= round(accuracy_score(y_pred, y_train)*100,2)
print('the accuracy score for training data of our model is :', acc_score)


# In[28]:


y_pred = model.predict(x_test)
acc_score = round(accuracy_score(y_pred, y_test)*100,2)
print('the accuracy score of our model is :', acc_score)


# In[29]:


from sklearn import metrics
score = round(model.score(x_test, y_test)*100,2)
print('score of our model is :', score)


# In[30]:


class_report = classification_report(y_pred, y_test)
print('classification report of our model: ', class_report)


# In[31]:


# we have achieved a model with decent accuracy score


# In[ ]:




