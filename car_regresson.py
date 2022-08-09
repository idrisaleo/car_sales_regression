import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score



## Drop the missing in the target variable.

car_sales = pd.read_csv('car-sales-extended-missing-data.csv')
car_sales.isna().sum()

car_sales.dropna(subset='Price', inplace=True)

## Split into X and y.

X = car_sales.drop('Price', axis=1)
y = car_sales['Price']

## Split into train and test sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Fill in the missing variable using SimpleImputer.

mc_cols = ['Make', 'Colour']
odo_col = ['Odometer (KM)']
doors_col = ['Doors']

mc_fill = SimpleImputer(strategy='constant', fill_value='missing')
odo_fill = SimpleImputer(strategy='mean')
doors_fill = SimpleImputer(strategy='constant', fill_value=4) ## decided to go for the most frequent num

## Use ColumnTransformer to organise the missing variables.

impute = ColumnTransformer([('mc', mc_fill, mc_cols),
                           ('odo', odo_fill, odo_col),
                           ('doors', doors_fill, doors_col)])

## Use fit_transform on the X_train and transform on the X_test.

X_train_trans = impute.fit_transform(X_train)
X_test_trans = impute.transform(X_test)

## Make a new dataframe for each of the X_train and X_test, ensuring it fits with the original df.

X_train_df = pd.DataFrame(X_train_trans, columns=['Make', 'Colour', 'Odometer (KM)', 'Doors'])
X_test_df = pd.DataFrame(X_test_trans, columns=['Make', 'Colour', 'Odometer (KM)', 'Doors'])

## Identify the variables u would like to finally transform to carry out your analysis.

var_to_trans = ['Make', 'Colour', 'Doors'] ## doors because its in float, i want all ints....

## Instantiate ur OneHotEncoder.

One_Hot = OneHotEncoder()

## use columnstranformer together with the onehotencoded, the variables u want tranformed

transform = ColumnTransformer([('One_Hot', One_Hot, var_to_trans)], remainder='passthrough')

## do a fit_transform for the x_train, and transform for the x_test.

final_X_train = transform.fit_transform(X_train_df)
final_X_test = transform.transform(X_test_df)

## instantiate the regression model and carry out your analysis

np.random.seed(0)
model = RandomForestRegressor()
model.fit(final_X_train, y_train)
model.score(final_X_test, y_test)
