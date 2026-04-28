import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import os

path = r"c:\IAIProject_AG"

# We can re-use the clean data since it has Latitude, Longitude, Date, Time, etc.
# Wait, we need the NDWI, LST, Rain extracted. The latest cells show they merge the extracted df.
# A fast way is just read the base accidents_clean and then extract values or look if we can extract spatial features.
print("Reading base accidents_clean")
df = pd.read_csv(os.path.join(path, "accidents_clean.csv"))

print(df.columns)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['dayofweek'] = df['Date'].dt.dayofweek
df['hour'] = pd.to_numeric(df['Hour'], errors='coerce')

df['risk'] = (df['Accident_Severity'] <= 2).astype(int)
print("Risk class balance:\n", df['risk'].value_counts(normalize=True))

# Fast spatial features
features = ['Latitude', 'Longitude', 'hour', 'month', 'dayofweek', 'Speed_limit']

train = df[df['year'] != 2022].copy()
test = df[df['year'] == 2022].copy()

X_train = train[features]
y_train = train['risk']

X_test = test[features]
y_test = test['risk']

clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, colsample_bytree=0.8, subsample=0.8, random_state=42)
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', callbacks=[lgb.early_stopping(50)])

preds = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)
print(f"LGBM AUC (with Lat/Lon): {auc}")
