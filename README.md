This code loads credit card fraud data, explores it, visualizes it, trains a random forest classifier model to predict fraud transactions and evaluates the model's performance. We load credit card transaction data, explore missing values, class distribution and visualize. Then we split data into train and test sets, standardize features, train a random forest classifier model and evaluate the model using accuracy, confusion matrix and classification report. From the graph we see there are significantly more valid transactions than fraudulent transactions and the outlier fraction is low, indicating data imbalance.

In general:
-> Models tend to be biased toward the majority class
-> Models may not learn enough from the minority class

To combat this we must:
-> Oversample the minority class (add duplicates)
-> Undersample the majority class (remove data points)

![image](https://github.com/KshitijShresth29/CreditCardFraudDetection/assets/145615126/ae89908a-862a-4850-9ec0-b4b06388df43)
