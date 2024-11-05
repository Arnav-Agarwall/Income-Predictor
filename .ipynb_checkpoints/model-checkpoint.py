import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('adult.brl.csv', na_values=" ?")


label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


x = data.drop("income", axis=1)
y = data["income"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=63)


model = RandomForestClassifier(n_estimators=150, random_state=63)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))