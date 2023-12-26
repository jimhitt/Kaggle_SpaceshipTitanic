import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# load data from data/train.csv into a pandas dataframe
df = pd.read_csv('data/train.csv')

# create X and y, parse Cabin into Deck, Number and Side, then split
y = df['Transported']

#copy X
X = df.copy(deep=True)
# split Cabin into Deck, Number and Side
X[['Cabin_Deck', 'Cabin_Number', 'Cabind_Side']] = X['Cabin'].str.split('/', expand=True)
# cast Cabin_number as int
X['Cabin_Number'] = X['Cabin_Number'].astype('float64')
# drop Cabin, PassengerId and Name
X.drop(['Cabin', 'PassengerId', 'Name', 'Transported'], axis=1, inplace=True)

# split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# create pipeline
# =============================================================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# generate numerical columns and categorical column lists from X_train datatypes   
numerical_columns = X_train.select_dtypes(include=['float64', 'int64', 'bool']).columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

# create inputers for numerical and categorical data
numerical_inputer = SimpleImputer(strategy='median')
categorical_inputer = SimpleImputer(strategy='most_frequent')

# create one hot encoder for categorical data
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

# create column transformer for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
            ('num', numerical_inputer, numerical_columns),
            ('cat', Pipeline(steps = [('imputer', categorical_inputer),
                                      ('onehot', one_hot_encoder)]), categorical_columns)
])

# assemble the pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# apply the pipeline to the training data
X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
# apply to validation data
X_val_preprocessed = preprocessing_pipeline.transform(X_val)

# print the shape of the training and validation data
print("Training data shape:", X_train_preprocessed.shape)
print("Validation data shape:", X_val_preprocessed.shape)