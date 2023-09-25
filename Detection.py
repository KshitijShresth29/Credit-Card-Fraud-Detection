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
