#!/usr/bin/env python
# coding: utf-8

# # TASK:1  Titanic Survival Prediction

# # Importing the package
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression


# # load the data frame from csv file to pandas dataframe

# In[2]:


m=pd.read_csv('tested.csv')
m


# # learning about the data set 

# In[3]:


m.head()


# In[4]:


m.tail()


# In[5]:


m.shape


# In[6]:


m.info()


# In[7]:


m.describe()


# In[8]:


m.isnull().sum()


# # Handling missing data

# In[9]:


m.drop(columns='Cabin',axis=1,inplace=True)


# In[10]:


Age=m['Age'].mean()


# In[11]:


m['Age'].fillna(Age,inplace=True)


# In[12]:


Fare=m['Fare'].mean()


# In[13]:


m['Fare'].fillna(Fare,inplace=True)


# In[14]:


m.info()


# In[15]:


m.isnull().sum()


# In[16]:


m


# # Data Visualization

# In[17]:


sns.set()


# In[18]:


sns.countplot(x='Sex',data=m)


# In[19]:


m['Survived'].value_counts()


# In[20]:


sns.countplot(x='Survived', data=m)


# In[21]:


sns.countplot (x='Sex', hue = 'Survived', data =m)


# In[22]:


sns.countplot(x='Pclass',data=m)


# In[23]:


m[['Survived', 'Sex']]


# In[24]:


m[['Survived', 'Pclass' ]]


# In[25]:


sns.countplot(x ='Pclass',hue= 'Survived', data=m)


# In[26]:


m['Sex'].value_counts()


# In[27]:


m['Embarked'].value_counts()


# # converting the categorical variables into numerical data

# In[28]:


m.replace({'Sex':{'male':0, 'female':1},'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)


# In[29]:


m


# In[30]:


# Now drop the columns which are irrelevant for the survival prediction, such as PassengerId, Name and Ticket


# In[31]:


m.drop(columns={'PassengerId','Name','Ticket'},axis=1, inplace=True)
m


# # separating features and target

# In[32]:


X = m.drop(columns='Survived', axis=1)
Y = m['Survived']
print(X)


# In[33]:


print(Y)


# # splitting the data into training and testing

# In[34]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
X_train.isnull().sum()


# # Model Training

# In[35]:


# we're using logistic regression model that uses binary classification for the p


# In[36]:


model = LogisticRegression()


# In[37]:


# we're using logistic regression model that uses binary classification for the predic


# In[40]:


model.fit(X_train,Y_train)


# # Model Evaluation

# In[56]:


#accuracy on taraining dataset
X_train_prediction=model.predict(X_train)
print(X_train_prediction)


# In[42]:


X_test_prediction = model.predict(X_test)


# In[43]:


# accuracy score for training data


# In[54]:


print(X_test_prediction)


# In[46]:


testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[47]:


print('Accuracy score of test data is : ',testing_data_accuracy)


# In[48]:


# precision score


# In[49]:


test_data_precision = precision_score(Y_test, X_test_prediction)


# In[50]:


print('test data precion is :', test_data_precision)


# In[ ]:




