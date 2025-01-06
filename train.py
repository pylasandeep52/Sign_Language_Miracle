# import pickle 
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import accuracy_score 


# with open("./ASL.pickle", "rb") as f:
#     dataset = pickle.load(f)


# data = np.asarray(dataset["dataset"])
# labels = np.asarray(dataset["labels"])

# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels, random_state=42)

# model = RandomForestClassifier()

# model1 =SVC(kernel='rbf',gamma=0.5,C=1.0)

# model.fit(X_train, y_train)

# model1.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# y_pred1 = model1.predict(X_test)

# score = accuracy_score(y_pred, y_test)
# print("accuracy of RandomForest is",score)

# score1 = accuracy_score(y_pred1, y_test)


# from sklearn.metrics import accuracy_score, precision_score, recall_score

# # Calculate accuracy, precision, and recall
# #accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')


# precision1 = precision_score(y_test, y_pred1, average='weighted')
# recall1 = recall_score(y_test, y_pred1, average='weighted')



# # Print the results
# #print(f'Accuracy: {accuracy:.2f}')
# print(f'Precision of Randomforest: {precision:.2f}')
# print(f'Recall: {recall:.2f}')

# print("accuracy of SVM is",score1)
# #print(f'Accuracy: {accuracy:.2f}')
# print(f'Precision of svm: {precision1:.2f}')
# print(f'Recall: {recall1:.2f}')


# #from sklearn import metrics

# #confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
# #print("confusion_matrix of RandomForest",confusion_matrix)
# from sklearn.linear_model import LogisticRegression
# model2= LogisticRegression()
# model2.fit(X_train, y_train)
# y_pred2 = model2.predict(X_test)
# score2 = accuracy_score(y_pred2, y_test)
# precision2 = precision_score(y_test, y_pred2, average='weighted')
# recall2 = recall_score(y_test, y_pred2, average='weighted')
# print("accuracy of Logistic Regression is",score2)
# print(f'Precision of Logistic Regression: {precision2:.2f}')
# print(f'Recall: {recall2:.2f}')



# with open("./ASL_model.p", "wb") as f:
#     pickle.dump({"model":model}, f)
import pickle 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load dataset from pickle file
with open("./ASL.pickle", "rb") as f:
    dataset = pickle.load(f)

# Convert dataset to numpy arrays
data = np.asarray(dataset["dataset"])
labels = np.asarray(dataset["labels"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels, random_state=42)

# Initialize models
model = RandomForestClassifier()
model1 = SVC(kernel='rbf', gamma=0.5, C=1.0)
model2 = LogisticRegression()

# Train models
model.fit(X_train, y_train)
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Predict using models
y_pred = model.predict(X_test)
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)

# Calculate accuracy, precision, and recall for Random Forest
accuracy_rf = accuracy_score(y_pred, y_test)
precision_rf = precision_score(y_test, y_pred, average='weighted')
recall_rf = recall_score(y_test, y_pred, average='weighted')

# Calculate accuracy, precision, and recall for SVM
accuracy_svm = accuracy_score(y_pred1, y_test)
precision_svm = precision_score(y_test, y_pred1, average='weighted')
recall_svm = recall_score(y_test, y_pred1, average='weighted')

# Calculate accuracy, precision, and recall for Logistic Regression
accuracy_lr = accuracy_score(y_pred2, y_test)
precision_lr = precision_score(y_test, y_pred2, average='weighted')
recall_lr = recall_score(y_test, y_pred2, average='weighted')

# Print results
print("Accuracy of RandomForest:", accuracy_rf)
print(f'Precision of RandomForest: {precision_rf:.2f}')
print(f'Recall of RandomForest: {recall_rf:.2f}')

print("Accuracy of SVM:", accuracy_svm)
print(f'Precision of SVM: {precision_svm:.2f}')
print(f'Recall of SVM: {recall_svm:.2f}')

print("Accuracy of Logistic Regression:", accuracy_lr)
print(f'Precision of Logistic Regression: {precision_lr:.2f}')
print(f'Recall of Logistic Regression: {recall_lr:.2f}')

# Save the Random Forest model using pickle
with open("./ASL_modelSVC.p", "wb") as f:
    pickle.dump({"model1": model1}, f)

# Plotting the comparison graph
labels = ['Accuracy', 'Precision', 'Recall']
random_forest_scores = [accuracy_rf, precision_rf, recall_rf]
svm_scores = [accuracy_svm, precision_svm, recall_svm]
logistic_regression_scores = [accuracy_lr, precision_lr, recall_lr]

x = np.arange(len(labels))  # The label locations
width = 0.25  # The width of the bars

fig, ax = plt.subplots(figsize=(6, 4))
rects1 = ax.bar(x - width, random_forest_scores, width, label='Random Forest', color='skyblue')
rects2 = ax.bar(x, svm_scores, width, label='SVM', color='lightgreen')
rects3 = ax.bar(x + width, logistic_regression_scores, width, label='Logistic Regression', color='salmon')

# Add labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('Comparison of Random Forest, SVM, and Logistic Regression')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value annotations on bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

# Show the plot
plt.tight_layout()
plt.show()