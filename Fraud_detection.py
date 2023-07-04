#!/usr/bin/env python
# coding: ut-8
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

data =  pd.read_csv("transactions_train.csv")

data.head()


data.describe()

data.shape

missing_values = data.isnull().sum()


missing_values


# Count the occurrences of each class in a column
class_counts = data['isFraud'].value_counts()

# Print the total number for each class
print(class_counts)


class_counts = data['type'].value_counts()

# Print the total number for each class
print(class_counts)


# In[10]:


data = data.drop('step', axis=1)


# In[11]:


from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit the LabelEncoder on the 'type' column
label_encoder.fit(data['type'])

# Transform the 'type' column using the fitted LabelEncoder and replace the original column
data['type_encoded'] = label_encoder.transform(data['type']) + 1


# In[12]:


data.head()


# In[13]:


from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit the LabelEncoder on 'nameOrig' column
label_encoder.fit(data['nameOrig'])

# Transform 'nameOrig' column using the fitted LabelEncoder
data['nameOrig_encoded'] = label_encoder.transform(data['nameOrig'])

# Fit the LabelEncoder on 'nameDest' column
label_encoder.fit(data['nameDest'])

# Transform 'nameDest' column using the fitted LabelEncoder
data['nameDest_encoded'] = label_encoder.transform(data['nameDest'])


# In[14]:


# Separate the target variable ('isFraud') from the input features
X = data[['amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'nameOrig_encoded', 'nameDest_encoded', 'type_encoded']]
y = data['isFraud']

# Verify the shapes of the target and input feature datasets
print("Shape of y (target):", y.shape)
print("Shape of X (input features):", X.shape)


# In[15]:


from sklearn.model_selection import train_test_split

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Further split the train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Verify the shapes of the train, validation, and test sets
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)


# In[16]:


# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[17]:


# Print the model summary
model.summary()


# In[18]:


# Compile the model
learning_rate = 0.5
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[19]:


# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)


# In[20]:


# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=40,
                    callbacks=[early_stopping],
                    verbose=1)


# In[21]:


history.history


# In[22]:


fig = plt.figure()
plt.plot(history.history['loss'], color='blue', label='loss')
plt.plot(history.history['val_loss'], color='green', label='val_loss')
fig.suptitle('LOSS OF TRAIN AND VAL', fontsize=30)
plt.legend(loc="upper left")
plt.show()


# In[23]:


fig = plt.figure()
plt.plot(history.history['accuracy'], color='blue', label='accuracy')
plt.plot(history.history['val_accuracy'], color='green', label='val_accuracy')
fig.suptitle('ACCURACY OF TRAIN AND VAL', fontsize=30)
plt.legend(loc="upper left")
plt.show()


# In[24]:


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# In[25]:


#Save the model
model.save('Fraud_detection_model.h5')

