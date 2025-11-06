import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# Load local dataset
df = pd.read_csv("D:\MULTIPLE  DISEASE PREDICTION SYSTEM\datas\parkinsons.csv")

df.drop(['name'], axis=1, inplace=True)

X = df.drop(columns=['status'], axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

pickle.dump(model, open('models/parkinsons_model.sav', 'wb'))

print("✅ Parkinson’s model trained and saved successfully!")
