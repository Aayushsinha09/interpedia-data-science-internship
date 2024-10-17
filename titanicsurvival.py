# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the training dataset
df_train = pd.read_csv('train.csv')

# Data preprocessing for training data
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_train.drop('Cabin', axis=1, inplace=True)

# Convert categorical columns into numerical values
label_encoder = LabelEncoder()
df_train['Sex'] = label_encoder.fit_transform(df_train['Sex'])
df_train['Embarked'] = label_encoder.fit_transform(df_train['Embarked'])

# Drop irrelevant columns
df_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Define features and target variable for training
X_train = df_train.drop('Survived', axis=1)
y_train = df_train['Survived']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Load the test dataset
df_test = pd.read_csv('test.csv')

# Data preprocessing for test data
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df_test['Embarked'].fillna(df_test['Embarked'].mode()[0], inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)

# Convert categorical columns into numerical values
df_test['Sex'] = label_encoder.transform(df_test['Sex'])  # Use transform here
df_test['Embarked'] = label_encoder.transform(df_test['Embarked'])  # Use transform here

# Drop irrelevant columns
df_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Make predictions on the test data
predictions = model.predict(df_test)

# Create a DataFrame for results
results = pd.DataFrame({'PassengerId': df_test.index + 892, 'Survived': predictions})  # Adjust PassengerId
results.to_csv('submission.csv', index=False)

print("Predictions saved to 'submission.csv'.")
