# Fraud_Detection_model

In this Python script, I read a CSV file named "transactions_train.csv" from Kaggle into a pandas DataFrame and performed some analysis on the dataset.

I dropped the "step" column as it was deemed irrelevant to the analysis. Then, I applied label encoding to the 'type' column using the LabelEncoder from the sklearn library. This allowed me to create additional encoded columns for 'nameOrig' and 'nameDest'.

To split the data into training, validation, and test sets, I used the train_test_split function from sklearn.

Next, I defined a sequential model using TensorFlow's Keras API. The model consists of dense layers with ReLU activation functions, and dropout layers were added to prevent overfitting. For binary classification, I used the sigmoid activation function in the output layer.

The model was compiled with the Adam optimizer, setting the learning rate to 0.5, and the loss function was set to binary crossentropy since the goal is binary classification.

During training, early stopping was used as a callback to prevent overfitting. The model was trained using the training and validation data.

Finally, the model was evaluated on the test set and achieved an accuracy of 99.9%. It's worth noting that the ratio of fraudulent transactions to non-fraudulent transactions is 1 to 823. This means that for every fraudulent transaction, there are 823 non-fraudulent transactions, causing the model to mostly predict transactions as non-fraudulent.


The dataset can be gotten from: 

https://www.kaggle.com/datasets/bannourchaker/frauddetection

Acknowledgements
This work is part of the research project ”Scalable resource-efficient systems for big data analytics” funded
by the Knowledge Foundation (grant: 20140032) in Sweden.

Please refer to this dataset using the following citations:

PaySim first paper of the simulator:

E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. 2016Acknowledgements

This work is part of the research project ”Scalable resource-efficient systems for big data analytics” funded
by the Knowledge Foundation (grant: 20140032) in Sweden.

Please refer to this dataset using the following citations:

PaySim first paper of the simulator:

E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. 2016
