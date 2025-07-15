# -------------------- CKD_Model_Training_Script --------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1 Read the CSV from the original location
df = pd.read_csv("https://github.com/Pavanfagare0023/Chronic_Kidney_Disease_Prediction/chronic_kidney_disease.csv")

# 2. Replace missing values represented by '?'
df.replace("?", np.nan, inplace=True)

# 3. Drop rows with too many missing values
df.dropna(thresh=10, inplace=True)

# 4. Handle missing values and encode categorical variables
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            df[col].fillna(df[col].mode()[0], inplace=True)
            df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

# 5. Drop ID column if present
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# 6. Set target column
target_col = 'classification'

# 7. Define features and target
X = df.drop(target_col, axis=1)
y = LabelEncoder().fit_transform(df[target_col])

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 10. Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 11. Save trained model
with open("https://github.com/Pavanfagare0023/Chronic_Kidney_Disease_Prediction/Dialysis_Status.pkl", 'wb') as file:
    pickle.dump(model, file)
print("Model saved as Dialysis_Status.pkl")
