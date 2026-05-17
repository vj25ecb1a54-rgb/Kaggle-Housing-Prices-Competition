import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

train = pd.read_csv("C:/Users/vikas/Downloads/train (1).csv")
test  = pd.read_csv("C:/Users/vikas/Downloads/test.csv")

train = train.drop(train[
    (train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)
].index)

def add_features(df):
    df = df.copy()

    def safe(col):
        return df[col] if col in df.columns else pd.Series(0, index=df.index)

    df['TotalSF']        = safe('1stFlrSF') + safe('2ndFlrSF') + safe('TotalBsmtSF')
    df['TotalPorchSF']   = (safe('OpenPorchSF') + safe('EnclosedPorch') +
                            safe('3SsnPorch')   + safe('ScreenPorch'))
    df['TotalIndoorSF']  = safe('GrLivArea') + safe('TotalBsmtSF')

    df['HouseAge']    = df['YrSold'] - df['YearBuilt']
    df['RemodelAge']  = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['GarageAge']   = df['YrSold'] - safe('GarageYrBlt').replace(0, np.nan)

    df['TotalBaths']  = (safe('FullBath') + 0.5 * safe('HalfBath') +
                         safe('BsmtFullBath') + 0.5 * safe('BsmtHalfBath'))

    df['QualCond']    = df['OverallQual'] * df['OverallCond']
    df['QualSF']      = df['OverallQual'] * df['TotalSF']
    df['QualGrLiv']   = df['OverallQual'] * safe('GrLivArea')

    df['HasPool']      = (safe('PoolArea')    > 0).astype(int)
    df['HasGarage']    = (safe('GarageArea')  > 0).astype(int)
    df['HasBsmt']      = (safe('TotalBsmtSF') > 0).astype(int)
    df['HasFireplace'] = (safe('Fireplaces')  > 0).astype(int)
    df['Has2ndFloor']  = (safe('2ndFlrSF')    > 0).astype(int)

    return df

train = add_features(train)
test  = add_features(test)

num_features = [
    "LotArea", "LotFrontage",
    "OverallQual", "OverallCond", "QualCond",
    "YearBuilt", "YearRemodAdd", "HouseAge", "RemodelAge", "IsRemodeled",
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "GrLivArea",
    "TotalSF", "TotalIndoorSF", "TotalPorchSF",
    "QualSF", "QualGrLiv",
    "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "TotalBaths",
    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageAge", "GarageCars", "GarageArea",
    "WoodDeckSF", "OpenPorchSF",
    "HasPool", "HasGarage", "HasBsmt", "HasFireplace", "Has2ndFloor",
]

cat_features = [
    "MSZoning", "Street", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType",
    "ExterQual", "ExterCond", "Foundation",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "Heating", "HeatingQC", "CentralAir", "Electrical",
    "KitchenQual", "Functional", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PavedDrive", "SaleType", "SaleCondition",
]

all_features = [f for f in num_features + cat_features
                if f in train.columns and f in test.columns]

X_train = train[all_features].copy()
X_test  = test[all_features].copy()
y_train = np.log1p(train["SalePrice"])

cat_cols_present = [c for c in cat_features if c in all_features]
num_cols_present = [c for c in num_features if c in all_features]

for col in num_cols_present:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col]  = X_test[col].fillna(median)

for col in cat_cols_present:
    X_train[col] = X_train[col].fillna("None").astype(str)
    X_test[col]  = X_test[col].fillna("None").astype(str)

    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined)

    X_train[col] = le.transform(X_train[col])
    X_test[col]  = le.transform(X_test[col])

print(f"Features: {X_train.shape[1]} | Train rows: {X_train.shape[0]}")

xgb = XGBRegressor(
    n_estimators     = 2000,
    learning_rate    = 0.02,
    max_depth        = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
)

lgb = LGBMRegressor(
    n_estimators      = 2000,
    learning_rate     = 0.02,
    max_depth         = 4,
    num_leaves        = 31,
    subsample         = 0.8,
    colsample_bytree  = 0.7,
    min_child_samples = 20,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    verbose           = -1,
    n_jobs            = -1,
)

gbr = GradientBoostingRegressor(
    n_estimators  = 2000,
    learning_rate = 0.02,
    max_depth     = 4,
    subsample     = 0.8,
    max_features  = "sqrt",
    random_state  = 42,
)

stacker = StackingRegressor(
    estimators      = [("xgb", xgb), ("lgb", lgb), ("gbr", gbr)],
    final_estimator = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
    cv              = 5,
    passthrough     = True,
    n_jobs          = -1,
)

print("Running 5-fold CV … (takes a few minutes)")
cv_scores = cross_val_score(
    stacker, X_train, y_train,
    cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
)
print(f"CV RMSE (log scale): {-cv_scores.mean():.5f} ± {cv_scores.std():.5f}")

stacker.fit(X_train, y_train)
final_preds = np.expm1(stacker.predict(X_test))

submission = pd.DataFrame({"Id": test["Id"], "SalePrice": final_preds})
submission.to_csv("submission11.csv", index=False)
