import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("D:\MULTIPLE  DISEASE PREDICTION SYSTEM\datas\diabetes.csv")

# Split into features and target
X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('models/diabetes_model.sav', 'wb'))

print("✅ Diabetes model trained and saved as models/diabetes_model.sav")
