import pickle

import pandas as pd

import catboost as ctb
import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score


class NumberTaker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class ExperienceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[X[self.column_name] == '<1', self.column_name] = 0
        X.loc[X[self.column_name] == '>20', self.column_name] = 21
        X = X.astype(int)
        return X

    
class NumpyToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.column_names)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Обучить и оценить модель.
    """
    model.fit(X_train, y_train, classifier__verbose=False)
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    
    return (model, {'f1': [f1], 'roc-auc': [roc], 'precision': [prec], 'recall': [rec]})


if __name__ == '__main__':
    # Read data
    df = pd.read_csv("data/aug_train.csv")
    # Target needs to be int
    df['target'] = df['target'].astype(int)
    
    # Model features
    num_feats = df.select_dtypes('number').drop(columns='target').columns.drop('enrollee_id')
    cat_feats = df.select_dtypes('object').columns

    # Impute missing values
    num_imputer = Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ])
    cat_imputer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ])
    imputers = ColumnTransformer([
        ('num_imputer', num_imputer, num_feats),
        ('cat_imputer', cat_imputer, cat_feats),
    ])

    # Transforming features
    num_transformer = Pipeline([
        ('nums', NumberTaker())
    ])
    cat_transformer = Pipeline([
        ('ohe', OneHotEncoder(drop='first', sparse=False))
    ])
    experience_transformer = Pipeline([
        ('experience_transform', ExperienceTransformer('experience'))
    ])
    city_transformer = Pipeline([
        ('city_transform', ce.cat_boost.CatBoostEncoder())
    ])
    transformers = ColumnTransformer([
        ('num_transformer', num_transformer, num_feats),
        ('cat_transformer', cat_transformer, cat_feats.drop(['experience', 'city'])),
        ('experience_transformer', experience_transformer, ['experience']),
        ('city_transformer', city_transformer, ['city']),
    ])
    
    # Putting together all preprocessing steps
    preprocessing = Pipeline([
        ('imputers', imputers),
        ('numpy_to_df', NumpyToDataFrame(num_feats.tolist() + cat_feats.tolist())),
        ('transforms', transformers)
    ])
    
    # Model
    clf = ctb.CatBoostClassifier()
    
    # Making a pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('classifier', clf),
    ])
    
    # Train (and evaluate) the model
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']),
                                                        df['target'],
                                                        test_size=0.2,
                                                        random_state=42)
    model, model_metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
    
    # Save the model (pickle it)
    with open("../models/ctb_clf.pkl", "wb") as file:
        model.__module__ = 'run_server'
        pickle.dump(model, file)
