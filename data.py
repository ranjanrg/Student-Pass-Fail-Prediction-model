import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# creating the dataset
data = {
  'hours' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  'result' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# separation of dataset

x = df[['hours']]
y = df['result']

# split data for trian and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

# creating the model

model = LogisticRegression()

# training the model
model.fit(x_train, y_train)

# predicting the model

prediction = model.predict(x_test)

print("Predicted value: ", prediction)
print("Actual value: ", y_test.values)

# accuracy
accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)

# finding the probablity

prob = model.predict_proba(pd.DataFrame({'hours': [4]}))

print(prob)