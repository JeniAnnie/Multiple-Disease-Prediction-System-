import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Use your local heart dataset
df = pd.read_csv("D:\MULTIPLE  DISEASE PREDICTION SYSTEM\datas\heart.csv")
import pandas as pd



X = df.drop(columns='target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pickle.dump(model, open('models/heart_disease_model.sav', 'wb'))

print("✅ Heart Disease model trained and saved successfully!")
