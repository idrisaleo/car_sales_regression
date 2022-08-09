# car_sales_regression

This is a regression analysis charged with understanding the contributory factors to the price of cars sold. 


1. Drop the missing in the target variable.
2. Split into X and y.
3. Split into train and test sets.
4. Fill in the missing variable using SimpleImputer.
5. Use ColumnTransformer to organise the missing variables.
6. Use fit_transform on the X_train and transform on the X_test.
7. Make a new dataframe for each of the X_train and X_test, ensuring it fits with the original df.
8. Identify the variables u would like to finally transform to carry out your analysis.
9. instantiate ur OneHotEncoder.
10. use columnstranformer together with the onehotencoded, the variables u want tranformed
11. do a fit_transform for the x_train, and transform for the x_test.
12. instantiate the regression model and carry out your analysis
