MNIST Data Prediction
This repository contains a Python script for loading, preprocessing, and building classification models using the MNIST dataset. The script trains and evaluates both Naive Bayes and Logistic Regression classifiers.

Requirements
Python 3.x
Pandas
Numpy
Scikit-learn
Matplotlib
Installation
To install the necessary libraries, you can use pip:

sh
Copy code
pip install pandas numpy scikit-learn matplotlib
Usage
Follow the instructions below to run the script and understand its functionalities.

1. Load and Check Data
The script loads the MNIST data into a pandas DataFrame and prints the keys of the dataset.

python
Copy code
from sklearn.datasets import fetch_openml
import pandas as pd

MINST_ujjwal = fetch_openml('mnist_784', version=1, parser='auto')
df_ujjwal = pd.DataFrame(data=MINST_ujjwal.data, columns=MINST_ujjwal.feature_names)
df_ujjwal['target'] = MINST_ujjwal.target
print(MINST_ujjwal.keys())
2. Assign Data to Variables
The script assigns the data and target to numpy arrays and prints their types and shapes.

python
Copy code
import numpy as np

x_ujjwal = np.array(MINST_ujjwal.data)
y_ujjwal = np.array(MINST_ujjwal.target)

print(type(x_ujjwal))
print(type(y_ujjwal))

print(x_ujjwal.shape)
print(y_ujjwal.shape)
3. Visualize Sample Digits
The script visualizes sample digits using Matplotlib.

python
Copy code
import matplotlib as mlp
import matplotlib.pyplot as plt

some_digit12 = x_ujjwal[3]
some_digit13 = x_ujjwal[8]
some_digit14 = x_ujjwal[1]

some_digit12_image = some_digit12.reshape(28, 28)
plt.imshow(some_digit12_image, cmap=mlp.cm.binary)

some_digit13_image = some_digit13.reshape(28, 28)
plt.imshow(some_digit13_image, cmap=mlp.cm.binary)

some_digit14_image = some_digit14.reshape(28, 28)
plt.imshow(some_digit14_image, cmap=mlp.cm.binary)
4. Pre-process the Data
The script changes the type of the target variable and transforms it into five classes.

python
Copy code
y_ujjwal = y_ujjwal.astype(np.uint8)

y_ujjwal[np.where((y_ujjwal >= 0) & (y_ujjwal <= 1))] = 0
y_ujjwal[np.where((y_ujjwal >= 2) & (y_ujjwal <= 3))] = 1
y_ujjwal[np.where((y_ujjwal >= 4) & (y_ujjwal <= 5))] = 2
y_ujjwal[np.where((y_ujjwal >= 6) & (y_ujjwal <= 7))] = 3
y_ujjwal[np.where((y_ujjwal >= 8) & (y_ujjwal <= 9))] = 4
5. Split the Data
The script splits the data into training and testing sets.

python
Copy code
x_train, x_test, y_train, y_test = x_ujjwal[:55000], x_ujjwal[55000:], y_ujjwal[:55000], y_ujjwal[55000:]
6. Train and Evaluate Models
The script trains and evaluates both Naive Bayes and Logistic Regression classifiers.

Naive Bayes
python
Copy code
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

NB_clf_ujjwal = GaussianNB()
NB_clf_ujjwal.fit(x_train, y_train)

scores = cross_val_score(NB_clf_ujjwal, x_train, y_train, cv=3)
print("3-fold cross-validation scores:", scores)
print("Average accuracy:", scores.mean())

test_accuracy = NB_clf_ujjwal.score(x_test, y_test)
print("Test data accuracy:", test_accuracy)

y_pred = NB_clf_ujjwal.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ", cm)
Logistic Regression
python
Copy code
from sklearn.linear_model import LogisticRegression

LR_clf_ujjwal_lb
I dont want codee, I just want read me file
ChatGPT
Sure, here is a sample README file with instructions on how to use your MNIST data prediction script:

MNIST Data Prediction
This repository contains a Python script for loading, preprocessing, and building classification models using the MNIST dataset. The script trains and evaluates both Naive Bayes and Logistic Regression classifiers.

Requirements
Python 3.x
Pandas
Numpy
Scikit-learn
Matplotlib
Installation
To install the necessary libraries, use pip:

sh
Copy code
pip install pandas numpy scikit-learn matplotlib
Usage
Follow the steps below to run the script and understand its functionalities.

Load and Check Data
Load the MNIST data into a pandas DataFrame.
Print the keys of the dataset.
Assign Data to Variables
Assign the data to a numpy array x_ujjwal.
Assign the target to a numpy array y_ujjwal.
Print the types and shapes of x_ujjwal and y_ujjwal.
Visualize Sample Digits
Create variables some_digit12, some_digit13, and some_digit14 to store sample values from x_ujjwal.
Use imshow method from Matplotlib to plot the values of these variables.
Pre-process the Data
Change the type of y_ujjwal to unit8.
Transform the target variable y_ujjwal to five classes:
0-1 → 0
2-3 → 1
4-5 → 2
6-7 → 3
8-9 → 4
Split the Data
Split the data into training and testing sets:
First 55,000 records for training.
Last 15,000 records for testing.
Train and Evaluate Models
Naive Bayes
Train a Naive Bayes classifier named NB_clf_ujjwal.
Use 3-fold cross-validation to validate the training process.
Evaluate the model on the test data and generate a confusion matrix.
Use the classifier to predict the sample digits some_digit12, some_digit13, and some_digit14.
Logistic Regression
Train a Logistic Regression classifier named LR_clf_ujjwal.
Train the classifier using two solvers: “lbfgs” and “saga”, with max_iter set to 900 and tol set to 0.1.
Compare the results of the two solvers.
Use 3-fold cross-validation on the training data.
Evaluate the model on the test data and note the regularization parameters.
Generate the accuracy matrix, precision, and recall of the model.
Use the classifier to predict the sample digits some_digit12, some_digit13, and some_digit14.
Analysis
Write a paragraph showing your conclusions regarding the training process in terms of bias and variance.
Carry out research on the differences between the “lbfgs” and “saga” solvers, and relate this to the size and dimension of the dataset.
Note the results of your research in your analysis report.
Make sure to replace any placeholder text with the actual results or additional details specific to your project as needed.
