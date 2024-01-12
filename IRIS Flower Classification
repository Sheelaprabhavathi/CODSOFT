#!/usr/bin/env python
# coding: utf-8

# # TASK:3  IRIS Flower Classification

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[4]:


iris_flower_file=pd.read_csv("IRIS.csv")
iris_flower_file


# In[5]:


iris_flower_file.shape


# In[6]:


iris_flower_file.info()


# In[7]:


iris_flower_file.describe()


# In[8]:


iris_flower_file.isnull().sum()


# In[9]:


iris_flower_file.describe()


# In[10]:


iris_flower_file['sepal_length'].hist()


# In[11]:


iris_flower_file['sepal_width'].hist()


# In[12]:


iris_flower_file['petal_length'].hist()


# In[13]:


iris_flower_file['petal_width'].hist()


# In[14]:


colors=['red','Black','teal']


# In[15]:


species=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


# In[16]:


for i in range(3):
    x=iris_flower_file[iris_flower_file['species']==species[i]]
    plt.scatter(x['sepal_length'],x['sepal_width'],c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[17]:


for i in range(3):
    x=iris_flower_file[iris_flower_file['species']==species[i]]
    plt.scatter(x['petal_length'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[18]:


for i in range(3):
    x=iris_flower_file[iris_flower_file['species']==species[i]]
    plt.scatter(x['sepal_length'],x['petal_length'],c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[19]:


for i in range(3):
    x=iris_flower_file[iris_flower_file['species']==species[i]]
    plt.scatter(x['sepal_width'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# In[20]:


numeric_columns=iris_flower_file.drop(columns='species')
corr=numeric_columns.corr()
fig,axis=plt.subplots(figsize=(5,5))
sns.heatmap(corr,annot=True,ax=axis,cmap='coolwarm')


# In[21]:


le=LabelEncoder()


# In[22]:


iris_flower_file['species']=le.fit_transform(iris_flower_file['species'])


# In[23]:


iris_flower_file.head(16)


# In[24]:


x=iris_flower_file.drop(columns='species')


# In[25]:


y=iris_flower_file['species']


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[27]:


LR=LogisticRegression()


# In[32]:


LR.fit(x_train,y_train)


# In[36]:


KNN=KNeighborsClassifier()
KNN.fit(x_train,y_train)


# In[37]:


DT=DecisionTreeClassifier()


# In[38]:


DT.fit(x_train,y_train)


# In[39]:


LR_accuracy=LR.score(x_test,y_test)*100
KNN_accuracy=KNN.score(x_test,y_test)*100
DT_accuracy=DT.score(x_test,y_test)*100


# In[40]:


print(f"Accuracy by using Logistic Regression: {LR_accuracy}%")


# In[41]:


print(f"Accuracy by using K Nearest Neighbors Algorithm: {KNN_accuracy}%")


# In[42]:


print(f"Accuracy by using Decision Tree Classifier: {DT_accuracy}%")


# In[ ]:




