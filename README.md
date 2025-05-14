# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VINOTHKUMAR R
RegisterNumber:  212224040361
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("/content/Employee.csv")
data.head()
```
```
data.tail()
```
```
data.isnull().sum()
```
```
data.info()
```
```
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years","salary"]]
x
```
```
y=data["left"]
y
```
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=100)
dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
```
```
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree  
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```


## Output:
**Head Values**
![1](https://github.com/user-attachments/assets/9601c933-ee1e-4d4c-a06a-27ae88732dba)

**Tail Values**
![2](https://github.com/user-attachments/assets/17bd87d9-25af-4caf-821e-473686879c35)


**Sum-Null Values**

![3](https://github.com/user-attachments/assets/6f5d12be-fa1f-4c56-b68f-aa34a412b131)

**DataInfo**
![4](https://github.com/user-attachments/assets/f665d9ec-b392-4a12-9705-34e3fd392280)


**Value Count in Left Column**

![5](https://github.com/user-attachments/assets/83f0c163-bc8f-4353-83e3-9bc834d91cb4)

**X-Values**

![6](https://github.com/user-attachments/assets/922137b6-fc64-463a-9c46-24ef4d1090c5)


**Y-Values**

![7](https://github.com/user-attachments/assets/ad33fe22-fa10-41b5-9ad4-266c453df48c)


**Training The Model**
![8](https://github.com/user-attachments/assets/0b06f327-0d0b-46da-999f-1bb015da1bf9)

**Accuracy AND Data Prediction**
![9](https://github.com/user-attachments/assets/e0677bce-7f85-4c36-afe2-2126e6ea0e4d)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
