# import numpy as np
# import pandas as pd

# df = pd.read_csv('fraud.csv')

# df.drop(columns=['nameOrig', 'nameDest'], axis=1)

# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()

# df['type'] = le.fit_transform(df['type'])

# X, y = df.loc[:, df.columns != 'isFraud'], df['isFraud']

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()

# # Keep only numeric columns
# X_train_numeric = X_train.select_dtypes(include=['int64', 'float64'])
# X_train = sc.fit_transform(X_train_numeric)

# X_test_numeric = X_test.select_dtypes(include=['int64', 'float64'])
# X_test = sc.transform(X_test_numeric)

# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state= 0)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print("Accuracy of the model:", metrics.accuracy_score(y_test, y_pred))

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load and clean data
df = pd.read_csv('fraud.csv')
df.drop(columns=['nameOrig', 'nameDest'], inplace=True)
df['type'] = preprocessing.LabelEncoder().fit_transform(df['type'])

# Feature and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scale numeric features
sc = StandardScaler()
X_train = sc.fit_transform(X_train.select_dtypes(include=['int64', 'float64']))
X_test = sc.transform(X_test.select_dtypes(include=['int64', 'float64']))

# Train model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# we can use it also as an alternative for Logistic Regression

# from sklearn.naive_bayes import GaussianNB
# from sklearn import metrics

# gnb = GaussianNB()
# gnb.fit(X_train, y_train)

# y_pred = gnb.predict(X_test)
# print("Accuracy of the model:", metrics.accuracy_score(y_test, y_pred))



# Predict and evaluate
y_pred = classifier.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1]))
