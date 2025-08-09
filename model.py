import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("Prawndataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Check for non-numeric values in target column
df = df[pd.to_numeric(df['Prawn_Count'], errors='coerce').notna()]  # Removes 'Unpredictable' or other strings

# Convert Prawn_Count to float
df['Prawn_Count'] = df['Prawn_Count'].astype(float)

# Encode categorical feature (Season)
label_encoder = LabelEncoder()
df['Season'] = label_encoder.fit_transform(df['Season'])

# Define input (X) and output (y)
X = df[['No_of_Prawns', 'Age_of_Pond', 'Food_Intake', 'Season']]
y = df['Prawn_Count']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE: {mae:.2f}")

# Save model & encoder
with open("prawn_model.pkl", "wb") as f:
    pickle.dump((model, label_encoder), f)
