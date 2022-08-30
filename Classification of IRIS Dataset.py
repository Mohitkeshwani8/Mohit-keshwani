#!/usr/bin/env python
# coding: utf-8

# # Classification of IRIS Dataset

# ## Importing IRIS Dataset

# In[134]:


from sklearn.datasets import load_iris
iris= load_iris()


# In[135]:


iris


# ## Keys

# In[136]:


print("Keys of iris_dataset:\n", iris.keys())


# ## IRIS Describing

# In[137]:


print(iris['DESCR'][:193] + "\n...")


# In[138]:


print("Target names:", iris['target_names'])


# In[139]:


print("Feature names:\n", iris_dataset['feature_names'])


# In[140]:


print("Data type:", type(iris['data']))


# In[141]:


print("Data shape:", iris['data'].shape)


# In[142]:


print(iris['filename'] + "\n...")


# ## Let's view the data

# In[143]:


print("First five rows of data:\n", iris_dataset['data'][:5])


# In[144]:


## Now as we have view of the data we can now train, test the data


# In[145]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[146]:


X_train.shape


# In[147]:


X_train


# In[148]:


X_test.shape


# In[149]:


X_test


# In[150]:


y_train


# In[151]:


y_test


# In[152]:


import pandas as pd
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)


# In[153]:


pip install mglearn


# In[154]:


import mglearn


# ## Plotting the Histograms

# In[155]:


pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=1.5, cmap=mglearn.cm3)


# In[156]:


## Now applying the KNN model


# In[157]:


from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors = 1)


# In[158]:


Knn.fit(X_train,y_train)


# ## Making Prediction

# X_new = np.array([[5, 2.9, 1, 0.2]])
# print("X_new.shape: {}".format(X_new.shape))

# In[159]:


prediction = Knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
iris_dataset['target_names'][prediction]))


# In[160]:


Knn = KNeighborsClassifier(n_neighbors = 1)
Knn.fit(X_train,y_train)
y_pred = Knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))


# ## Evaluation of Model

# In[161]:


print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

