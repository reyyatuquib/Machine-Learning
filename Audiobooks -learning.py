#!/usr/bin/env python
# coding: utf-8

# In[11]:


## Create the machine learning algorithm


# In[12]:


##Import the relevant libraries


# In[3]:


import numpy as np
import tensorflow as tf


# In[4]:


##Data


# In[5]:


npz = np.load('Audiobooks_data_train.npz')

train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_validation.npz')
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_test.npz')
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)


# In[6]:


# Model (outline, optimizers,loss, early stopping and training)


# In[8]:


input_size = 10
output_size = 2
hidden_layer_size = 50

model = tf.keras.Sequential([
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation ='softmax') 
                            ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

batch_size = 100

max_epochs = 100

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(train_inputs,
         train_targets,
         batch_size = batch_size,
         epochs = max_epochs,
         callbacks =[early_stopping],
         validation_data=(validation_inputs, validation_targets),
         verbose = 2)


# In[9]:


## Test the models


# In[10]:


test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)


# In[14]:


print('\nTest loss:{0:2f}. Test accuracy:{1:.2f}%'.format (test_loss, test_accuracy*100))


# In[ ]:




