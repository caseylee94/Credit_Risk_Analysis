# Credit_Risk_Analysis

## Purpose

Predicting the risk of lending money through credit cards can be a tricky value to numerically uncover. Credit risk is an inherently unbalanced classification problem as typically the amount of good loans easily outnumber the risky loans. However, though good loans outnumber bad loans the risk is still high; just a few loans in large amounts that do not get paid back can significantly impact a credit company. To tackle a complex, multifaceted problem such as this, data analysts turn to machine learning. 

In this analysis, a credit card dataset from LendingClub, a peer-to-peer lending services company, will be oversampled and undersampled and ran through six different machine learning models. This will allow different models to be compared to find the best fit (highest accuracy) for predicting trends using this dataset as a training model. Using a combinatorial approach of under- and oversampling will offset the unbalanced representation of bad loans in the dataset and help to train a model that will accurately predict future loans, helping credit companies determine which loans are risky before the money is lent.

## Analysis and Results

Before the data could be trained on a machine learning model, the data needed to be preprocessed. All null values were dropped, all columns were converted to numerical values, and the target column (risk) was separated and converted into two columns, one for low risk and one for high risk. The data was separated into features and the target and then split into training and testing datasets using *sklearn*.

First we will discuss the oversampling methods. For this analysis, the *RandomOverSampler* from the *imblearn* library was imported and used to resample the training and testing data. The resampled data was trained on the logistic regression model and the balanced accuracy score, confusion matrix, and classfication report was generated using the *imbalanced_classification_report* from *imbalanced-learn*. A random state of 1 was used for each model to ensure consistency.

 ![RandomOversampling.png](/Resources/RandomOversampling.png)
 *Figure 1: The classification report for the random oversampler model*


## Summary
