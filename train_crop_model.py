import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset

df = pd.read_csv("Crop_recommendation.csv")

# Encode crop labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Save label encoder
joblib.dump(le, 'label_encoder.pkl')

# Split features and labels
X = df.drop('label', axis=1)
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'crop_model.pkl')
print("Model and LabelEncoder saved!")
