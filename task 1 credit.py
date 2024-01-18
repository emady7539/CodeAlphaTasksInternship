import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# TASK 1
# Load data
# Replace 'your_data.csv' with your actual csv file
df = pd.read_csv('your_data.csv')

# Assume the last column is the target variable 'creditworthy'
X = df.iloc[:, :-1]  # features
y = df.iloc[:, -1]  # target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Assess the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')