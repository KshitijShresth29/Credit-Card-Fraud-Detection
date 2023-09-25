import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset from the CSV file using pandas
data_url = "https://confrecordings.ams3.digitaloceanspaces.com/CreditCard_9820.csv"
data = pd.read_csv(data_url)

# Data Exploration and Preprocessing
# Check for missing values
missing_values = data.isnull().sum()

# Drop rows with missing values (you can also choose to impute)
data.dropna(inplace=True)

# Class Distribution Analysis
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid))

print("Number of Fraud Cases:", len(fraud))
print("Number of Valid Transactions:", len(valid))
print("Outlier Fraction:", outlier_fraction)

plt.figure(figsize=(10, 8))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution (0: Valid, 1: Fraud)')
plt.show()
# Feature Selection and Data Splitting
X = data.drop(['Class'], axis=1)
Y = data["Class"]
# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
