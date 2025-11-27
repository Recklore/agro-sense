import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Read the CSV
df = pd.read_csv("./data/Crop_recommendation.csv")

# 2. Separate features and target
X = df.drop("label", axis=1)
y = df["label"]

# 3. Train–test split (80–20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Random Forest model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# 5. Predictions on test set
y_pred = rf.predict(X_test)

# 6. Evaluate accuracy
acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", acc)

with open("./models/crop_model.pkl", "wb") as f:
    pickle.dump(rf, f)