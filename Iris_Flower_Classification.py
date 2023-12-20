#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION

#  ### TASK 1

# #### The Iris flower dataset encompasses three distinct species: setosa, versicolor, and virginica.
# • These species are discernible through specific measurements. Imagine
# possessing measurements of Iris flowers categorized by their distinct species.
# • The goal is to train a machine learning model capable of learning from these
# measurements and proficiently categorizing Iris flowers into their corresponding
# species.
# • Employ the Iris dataset to construct a model adept at classifying Iris flowers into
# distinct species based on their sepal and petal measurements.
# • This dataset serves as a prevalent choice for initial classification tasks, making it
# ideal for introductory learning experiences

# In[27]:


from IPython.display import Image
Image(url='https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png', width=850)
     


# ## IMPORTING LIBRARIES

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the Dataset

# In[29]:


iris = pd.read_csv('IRIS.csv')
iris.head()


# In[30]:


# Rename the complex columns name
iris= iris.rename(columns={'SepalLengthCm':'Sepal_Length',
                           'SepalWidthCm':'Sepal_Width',
                           'PetalLengthCm':'Petal_Length',
                           'PetalWidthCm':'Petal_Width'})


# In[31]:


iris.head()


# In[32]:


# checking null values
iris.isnull().sum()


# In[33]:


# checking if the data is biased or not
iris ['species'].value_counts()
     


# In[34]:


# checking statistical features
iris.describe()


# ## Visualization

# ## Scatterplot

# In[35]:



sns.FacetGrid(iris, hue="species",height=6).map(plt.scatter,"petal_length","sepal_width").add_legend()


# ## Pairplot

# In[36]:


# visualize the whole dataset
sns.pairplot(iris[['sepal_length','sepal_width','petal_length','petal_width','species']], hue="species",diag_kind='kde')
     


# ## SEPARATING INPUT COLUMNS AND THE OUTPUT COLUMNS

# In[59]:


# Separate features and target
data=iris.values

# slicing the matrices
X=data[:,0:4]
Y=data[:,4]


# In[53]:


print(X.shape)
print(X)
     


# In[60]:


print(Y.shape)
print(Y)
     


# ## SPLITTING DATA INTO TRAINING AND TESTING

# In[61]:


#split the data to train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.2)
    
   


# In[62]:


print(X_train.shape)
print(X_train)


# In[63]:


print(y_test.shape)
print(y_test)


# In[65]:



print(X_test.shape)
print(X_test)
     


# In[66]:


print(Y.shape)
print(Y)
     


# ## SPLITTING DATA INTO TRAINING AND TESTING

# In[67]:


# split the data to train and test dataset
     

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.2)


# In[68]:


print(X_train.shape)
print(X_train)
     


# In[69]:


print(y_test.shape)
print(y_test)


# In[70]:


print(X_test.shape)
print(X_test)
     


# In[71]:


print(y_train.shape)
print(y_train)
     


# ## MODEL 1: SUPPORT VECTOR MACHINE ALGORITHM

# In[73]:


from sklearn.svm import SVC

model_svc=SVC()
model_svc.fit(X_train,y_train)
     


# In[74]:


prediction1 = model_svc.predict(X_test)

#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction1))
     


# ## MODEL 2: LOGISTIC REGRESSION
# 

# In[76]:


# converting categorical variables into numbers
flower_mapping = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
iris['species']=iris['species'].map(flower_mapping)
     


# In[77]:


iris.head()
     


# In[78]:


iris.tail()


# In[79]:


# preparing inputs and outputs
X=iris [['sepal_length','sepal_width','petal_length','petal_width']].values
y= iris[['species']].values


# In[80]:


# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X,y)
     


# In[81]:


# accuracy
     

model.score(X,y)
     


# In[82]:


# make prediction for all 150 species in dataset
     

expected = y
predicted = model.predict(X)
predicted


# In[83]:


# summarize the fit of the model

from sklearn import metrics
     


# In[84]:


print(metrics.classification_report(expected, predicted))


# In[85]:


# confusion metrics
print(metrics.confusion_matrix(expected, predicted))
     


# ## MODEL3: DECISION TREE CLASSIFIER

# In[86]:


from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train, y_train)
     


# In[87]:


prediction3= model_svc.predict(X_test)

#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction3))
     


# ## New data for prediction

# In[88]:


# New data for prediction
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])

# Predicting the sizes of the iris flowers
predicted_sizes = model.predict(X_new)

# Output the predicted sizes
print(predicted_sizes)


# In[ ]:




