import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# TASK4
# Thank you CODE ALPHA for this amazing journey I hope I continue and meet your expectations and I refered some friends to you  and still there are more.
# Load dataset
# Replace 'medical_data.csv' with your actual file path
df = pd.read_csv('medical_data.csv')

# Assume the last column is the target variable (disease present: 1, not present: 0)
X = df.iloc[:, :-1]  # features (symptoms, patient history)
y = df.iloc[:, -1]  # target variable (disease)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print('Model accuracy:', accuracy)