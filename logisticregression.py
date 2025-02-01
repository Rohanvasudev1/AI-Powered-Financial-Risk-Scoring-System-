import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib as plt

df = pd.read_csv("data/application_train.csv")
credit_risk_columns = ["TARGET",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "DAYS_EMPLOYED", "OCCUPATION_TYPE", "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH", "CODE_GENDER", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "NAME_FAMILY_STATUS", "OWN_CAR_AGE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG",
    "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "TOTALAREA_MODE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_LAST_PHONE_CHANGE",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"
]
df_filtered = df[credit_risk_columns]
df_filtered.to_csv("filteredData.csv", index = False)

numeric_cols = df_filtered.select_dtypes(include=['number']).columns
df_filtered[numeric_cols] = df_filtered[numeric_cols].fillna(df_filtered[numeric_cols].mean())
df_filtered

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


categorical_cols = df_filtered.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))  
    label_encoders[col] = le
df_filtered

X = df_filtered.drop(columns=['TARGET'])
y = df_filtered['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train[X_train.select_dtypes(include=['number']).columns] = scaler.fit_transform(X_train[X_train.select_dtypes(include=['number']).columns])
X_test[X_test.select_dtypes(include=['number']).columns] = scaler.transform(X_test[X_test.select_dtypes(include=['number']).columns])

log_reg = LogisticRegression(max_iter=1000)  
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1] 

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
y_pred_proba