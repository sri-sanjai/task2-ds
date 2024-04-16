#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


titanic_data = pd.read_csv("titanic.csv")


# In[3]:


print(titanic_data.head())


# In[4]:


print(titanic_data.isnull().sum())


# In[7]:


median_age = titanic_data['Age'].median()
titanic_data['Age'].fillna(median_age, inplace=True)


# In[11]:


mode_embarked = titanic_data['Embarked'].mode()[0]
titanic_data['Embarked'].fillna(mode_embarked, inplace=True)


# In[12]:


print(titanic_data.columns)


# In[13]:


print(titanic_data.describe())


# In[14]:


print(titanic_data.describe(include=['O']))


# In[15]:


sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival Count by Gender')
plt.show()


# In[16]:


sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.title('Survival Count by Passenger Class')
plt.show()


# In[17]:


plt.figure(figsize=(10, 6))
sns.histplot(titanic_data['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[18]:


plt.figure(figsize=(10, 6))
sns.histplot(x='Age', hue='Survived', data=titanic_data, bins=20, kde=True)
plt.title('Survival Count by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[19]:


plt.figure(figsize=(10, 6))
sns.histplot(x='Fare', hue='Survived', data=titanic_data, bins=20, kde=True)
plt.title('Survival Count by Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()


# In[ ]:




