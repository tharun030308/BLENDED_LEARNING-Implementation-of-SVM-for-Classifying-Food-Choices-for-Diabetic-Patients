# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries, load the dataset, and select relevant features and target variable.
2.Split the dataset into training and testing sets and perform feature scaling using StandardScaler.
3.Create the SVM model, apply GridSearchCV for hyperparameter tuning, and train the model.
4.Predict the test data and evaluate the model using accuracy, classification report, and confusion matrix visualization.  

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)

features = ['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target = 'class'
x = data[features]
y = data[target]

x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



svm = SVC()

param_grid = {
    'C': [0.1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale','auto']
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train,y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
y_pred=best_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Name:Barath B")
print("Register Number:25009091")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="782" height="585" alt="image" src="https://github.com/user-attachments/assets/584a9517-3310-4b7e-9335-556e36722039" />
<img width="723" height="409" alt="image" src="https://github.com/user-attachments/assets/c06d0cfa-7b43-40a1-b59e-4a7ab427a2ed" />
<img width="748" height="562" alt="image" src="https://github.com/user-attachments/assets/b1c60f54-b396-4f57-a45b-9d1ac76f2026" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
