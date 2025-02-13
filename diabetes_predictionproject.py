import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r'C:\\Users\\gupta\\Downloads\\diabetes.csv')

# Define features and target variable
X = data.iloc[:, :-1]  # Select all columns except the last one as features
Y = data.iloc[:, -1]   # Select the last column as the target

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Print shapes of the dataset
print("Dataset Shapes:", X.shape, xtrain.shape, xtest.shape)

# Initialize and train the SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(xtrain, ytrain)

# Predict on training data
xtrain_prediction = classifier.predict(xtrain)
train_acc = accuracy_score(ytrain, xtrain_prediction)
print("Training Accuracy:", train_acc)

# Predict on test data
xtest_prediction = classifier.predict(xtest)
test_acc = accuracy_score(ytest, xtest_prediction)
print("Test Accuracy:", test_acc)

# Predict for a single input sample
a=int(input("Enter pregnencies:"))
a1=float(input("Enter Glucose level:"))
a2=float(input("Enter BloodPressure:"))
a3=int(input("Enter SkinThickness:"))
a4=float(input("Enter Insulin Level:"))
a5=float(input("Enter Body Mass Index(BMI):"))
a6=float(input("Enter DiabetesPedigreeFunction:"))
a7=int(input("Enter your age:"))

input_data = (a,a1,a2,a3,a4,a5,a6,a7)  # Ensure that this has the same number of features
input_array = np.asarray(input_data).reshape(1, -1)  # Convert to NumPy array and reshape
input_scaled = scaler.transform(input_array)  # Standardize input data

# Make prediction
prediction = classifier.predict(input_scaled)
print("Prediction:", prediction[0])  # Display result
if(prediction[0]==0):
    print('Person is not diabetic')
else:
    print('Person is diabetic')
