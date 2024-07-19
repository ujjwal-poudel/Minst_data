#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:31:37 2024

@author: ujjwalpoudel
"""

'''
Load and Check Data
'''

"""
1. Load the MINST data into a pandas dataframe named MINST_firstname where first
name is youname. (set the parser='auto)
"""

from sklearn.datasets import fetch_openml
import pandas as pd

MINST_ujjwal = fetch_openml('mnist_784', version=1, parser='auto')
df_ujjwal = pd.DataFrame(data=MINST_ujjwal.data, columns=MINST_ujjwal.feature_names)
df_ujjwal['target'] = MINST_ujjwal.target


"""
2. List the keys
"""
print(MINST_ujjwal.keys())

"""
3. Assign the data to a ndarray named X_firstname where firstname is your first name
"""
import numpy as np
x_ujjwal = np.array(MINST_ujjwal.data)

"""
4. Assign the target to a variable named y_firstname where firstname is your first name
"""
y_ujjwal = np.array(MINST_ujjwal.target)

"""
5. Print the types of X_firstname and y_firstname
"""
print(type(x_ujjwal))
print(type(y_ujjwal))

"""
6. Print the shape of X_firstname and y_firstname
"""
print(x_ujjwal.shape)
print(y_ujjwal.shape)

"""
7. Create three variables named as follows
# If your first name starts from “M” through “Z” name the variable
# some_digit12, some_digit13, some_digit14. Store in these variables the
# values from X_firstnameindexed 3,8,1 in order.
"""

some_digit12 = x_ujjwal[3]
some_digit13 = x_ujjwal[8]
some_digit14 = x_ujjwal[1]

# print(y_ujjwal[3])

"""
8. Use imshow method to plot the values of the three variables you defined in the
above point.Note the values in your Analysis report (written response).
"""
import matplotlib as mlp
import matplotlib.pyplot as plt

# Displaying the first digit
some_digit12_image = some_digit12.reshape(28, 28)
plt.imshow(some_digit12_image, cmap=mlp.cm.binary)

# Displaying the second digit
some_digit13_image = some_digit13.reshape(28, 28)
plt.imshow(some_digit13_image, cmap=mlp.cm.binary)

# Displaying the third image
some_digit14_image = some_digit14.reshape(28, 28)
plt.imshow(some_digit14_image, cmap=mlp.cm.binary)



'''
Pre-process the data
'''

"""
9. Change the type of y to unit8
"""
y_ujjwal = y_ujjwal.astype(np.uint8)

"""
10. The current target values range from 0 to 9 i.e. 10 classes. Transform the target
variable to five classes as follows:
    
a. Any digit between 0 and 1 inclusive should be assigned a target value of 0
b. Any digit between 2 and 3 inclusive should be assigned a target value of 1
c. Any digit between 4 and 5 inclusive should be assigned a target value of 2
d. Any digit between 6 and 7 inclusive should be assigned a target value of 3
e. Any digit between 8 and 9 inclusive should be assigned a target value of 4
"""
y_ujjwal[np.where((y_ujjwal >= 0) & (y_ujjwal <= 1))] = 0
y_ujjwal[np.where((y_ujjwal >= 2) & (y_ujjwal <= 3))] = 1
y_ujjwal[np.where((y_ujjwal >= 4) & (y_ujjwal <= 5))] = 2
y_ujjwal[np.where((y_ujjwal >= 6) & (y_ujjwal <= 7))] = 3
y_ujjwal[np.where((y_ujjwal >= 8) & (y_ujjwal <= 9))] = 4

"""
11. Print the frequencies of each of the five target classes and note it in your written
report inaddition provide a screenshot showing a bar chart.
"""
# Counting the frequency of each class
class_counts = np.unique(y_ujjwal, return_counts=True)[1]
print(class_counts)

# Printing the frequencies
print("Class frequencies:")
for i, count in enumerate(class_counts):
  print(f"Class {i}: {count}")

# Creating a bar chart
plt.bar(range(len(class_counts)), class_counts)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.title("Frequency of Each Target Class")
plt.show()

"""
12. Split your data into train, test. Assign the first 55,000 records for training and the last 15,000
records for testing. (Hint you don’t need sklearn train test as the data is already randomized)
"""
x_train, x_test, y_train, y_test = x_ujjwal[:55000], x_ujjwal[55000:], y_ujjwal[:55000], y_ujjwal[55000:]

'''
Build Classification
ModelsNaïve Bayes
'''

"""
13. Train a Naive Bayes classifier using the training data. Name the classifier NB_clf_firstname. . (Note this
is a multi-class problem make sure to check all the parameters and choose the correct ones)
"""
from sklearn.naive_bayes import GaussianNB

NB_clf_ujjwal = GaussianNB(priors=None, var_smoothing=1e-09)
NB_clf_ujjwal.fit(x_train, y_train)

"""
14. Use 3-fold cross validation to validate the training process, and note the results in
your written response. Also, note in your report the regularization parameters used
in the training process.
"""

from sklearn.model_selection import cross_val_score

# Defining the cross-validation strategy
cv = 3

# Evaluating the model using cross-validation
scores = cross_val_score(NB_clf_ujjwal, x_train, y_train, cv=cv)

# Printing the results
print("3-fold cross-validation scores:", scores)
print("Average accuracy:", scores.mean())

# Printing the regularization parameters
print("Regularization parameters:")
print("- Prior probabilities:", NB_clf_ujjwal.class_prior_)
print("- Variance smoothing:", NB_clf_ujjwal.var_smoothing)

"""
15. Use the model to score the accuracy against the test data, note the result in
your writtenresponse.
"""
test_accuracy = NB_clf_ujjwal.score(x_test, y_test)
print("Test data accuracy:", test_accuracy)

"""
16. Generate the confusion matrix.
"""
from sklearn.metrics import confusion_matrix
y_pred = NB_clf_ujjwal.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ", cm)

"""
17. Use the classifier to predict the three variables you defined in point 7 above. Note the
results in your written response and compare against the actual results.
"""

some_digits = [some_digit12, some_digit13, some_digit14]
predicted_labels = NB_clf_ujjwal.predict(some_digits)

# Printing the predicted results
print("Predicted labels for some_digit12, some_digit13, some_digit14:", predicted_labels)


"""
18. In your written analysis write a paragraph showing your conclusions with regard to
the training process in term of bias and variance.
"""
# Answer in written analysis



'''
Logistic regression
'''

"""
19. Train a Logistic regression classifier using the same training data. Name the classifier
LR_clf_firstname. (Note this is a multi-class problem make sure to check all the
parameters andset multi_class='multinomial').
"""

from sklearn.linear_model import LogisticRegression

# Defining the Logistic Regression classifier with multi_class set to 'multinomial'
LR_clf_ujjwal = LogisticRegression(multi_class='multinomial')

# Training the classifier on the training data
LR_clf_ujjwal.fit(x_train, y_train)


"""
20. Try training the classifier using two solvers first “lbfgs” then “Saga”. Set max_iter to
900 andtolerance to 0.1 in both cases.
"""
from sklearn.metrics import accuracy_score

# Training Logistic Regression with 'lbfgs' solver
LR_clf_ujjwal_lbfgs = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=2000,
    tol=0.1
)

# Fitting the data
LR_clf_ujjwal_lbfgs.fit(x_train, y_train)

lbfgs_train_accuracy = accuracy_score(y_train, LR_clf_ujjwal_lbfgs.predict(x_train))

lbfgs_test_accuracy = accuracy_score(y_test, LR_clf_ujjwal_lbfgs.predict(x_test))

print("LBFGS Solver:")
print("Training accuracy:", lbfgs_train_accuracy)
print("Test accuracy:", lbfgs_test_accuracy)

# Train Logistic Regression with 'saga' solver
LR_clf_ujjwal_saga = LogisticRegression(
    multi_class='multinomial',
    solver='saga',
    max_iter=900,
    penalty='l1',
    tol=0.1
)

LR_clf_ujjwal_saga.fit(x_train, y_train)

saga_train_accuracy = accuracy_score(y_train, LR_clf_ujjwal_saga.predict(x_train))
saga_test_accuracy = accuracy_score(y_test, LR_clf_ujjwal_saga.predict(x_test))
print("SAGA Solver:")
print("Training accuracy:", saga_train_accuracy)
print("Test accuracy:", saga_test_accuracy)


"""
21. Make sure you note the results in both cases in your written response, and note the
main differences in your written response. Don’t worry if one doesn’t converge your
research should explain why.
"""
# Answer in written response

"""
22. Carryout a research on the difference between the “lbfgs” and “Saga” solvers and
see how this applies to the results, relate that to the that size and dimension of the
dataset.
"""
# Answer in Written response

"""
23. Note the results of your research in your analysis report.
"""
# Answer in Written response

"""
24. Use 3-fold cross validation on the training data and note the results in your written response.
"""
# Defining the cross-validation strategy
cv = 3

# Evaluating the model using cross-validation
scores_logistic = cross_val_score(LR_clf_ujjwal_saga, x_train, y_train, cv=cv)

# Printing the results
print("3-fold cross-validation scores:", scores_logistic)
print("Average accuracy:", scores_logistic.mean())

"""
25. Use the model to score the accuracy against the test data, note the result in your
written response. Also, note in your report the regularization parameters used in the
training process.
"""
saga_train_accuracy = accuracy_score(y_train, LR_clf_ujjwal_saga.predict(x_train))
saga_test_accuracy = accuracy_score(y_test, LR_clf_ujjwal_saga.predict(x_test))

print("SAGA Solver with data:")
print("Training accuracy:", saga_train_accuracy)
print("Test accuracy:", saga_test_accuracy)

"""
26. Generate the Generate the accuracy matrix precision and recall of the model and
note them inyour written response.
"""
from sklearn.metrics import precision_score, recall_score

# Predicting using the trained Logistic Regression model
y_pred = LR_clf_ujjwal_saga.predict(x_test)

# Calculating the accuracy again
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculating the precision
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)

# Calculating the recall
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)

# Generating the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

"""
27. Use the classifier that worked from the above point to predict the three variables you
defined inpoint 7 above. Note the results in your written response and compare
against the actual results
"""

some_digits = [some_digit12, some_digit13, some_digit14]
predicted_labels_LR = LR_clf_ujjwal_saga.predict(some_digits)

print(predicted_labels_LR)

"""
28. In your written analysis write a paragraph showing your conclusions with regard to the
training process in term of bias and variance
"""
# Answer in written response