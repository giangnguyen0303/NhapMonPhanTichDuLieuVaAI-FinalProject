"""Gợi ý pipeline cho nhóm ML; fit chỉ trên training set (không fit toàn dữ liệu)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(numerical_features, categorical_features):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())])
    category = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num', numeric, numerical_features),
                              ('cat', category, categorical_features)])
