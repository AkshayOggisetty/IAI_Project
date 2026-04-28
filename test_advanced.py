import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import os

path = r"c:\IAIProject_AG"

df = pd.read_csv(os.path.join(path, "accidents_clean.csv"))
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['dayofweek'] = df['Date'].dt.dayofweek
df['hour'] = pd.to_numeric(df['Hour'], errors='coerce')

df['risk'] = (df['Accident_Severity'] <= 2).astype(int)

# Create a spatial grid rounding Lat and Lon to 2 decimal places (~1.1 km resolution)
df['lat_grid'] = df['Latitude'].round(2)
df['lon_grid'] = df['Longitude'].round(2)
df['grid_id'] = df['lat_grid'].astype(str) + "_" + df['lon_grid'].astype(str)

train = df[df['year'] != 2022].copy()
test = df[df['year'] == 2022].copy()

# Target encoding for the grid
grid_risk_mapping = train.groupby('grid_id')['risk'].mean()
train['grid_risk_encoded'] = train['grid_id'].map(grid_risk_mapping)

# Fill missing values in test set using the global mean
global_mean = train['risk'].mean()
test['grid_risk_encoded'] = test['grid_id'].map(grid_risk_mapping).fillna(global_mean)

features = ['Latitude', 'Longitude', 'hour', 'month', 'dayofweek', 'Speed_limit', 'grid_risk_encoded']

X_train = train[features]
y_train = train['risk']

X_test = test[features]
y_test = test['risk']

xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='auc')
xgb_model.fit(X_train, y_train)

preds = xgb_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)
print(f"XGBoost AUC with Spatial Target Encoding: {auc}")
