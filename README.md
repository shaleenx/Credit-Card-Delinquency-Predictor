# Credit-Card-Delinquency-Predictor
##Machine Learning Model for Predicting Credit Card Charge-Offs in the Next 2 Years

##Dependencies
1. Python v2.7.3
2. Pandas v0.17.1 or more
3. Numpy 1.10.1 or more
4. Sklearn 0.17
5. Statsmodels 0.6.1

##Usage
```
python modelbuilder.py
```

##Model Objective
The training dataset would contain data about credit card accounts that are currently Bucket 1 Delinquent.
The data set would contain a target flag which tells whether that account went into the Delinquency Bucket 4 in the next 9 months or not.
After training the model on the prescribed data set, the model would be able to predict the probability of a credit card account to go into the Delinquency Bucket 4, given that it is already in Delinquency Bucket 1.

##Note
One must train the model on a dataset similar to the one used, the details of which cannot be shared publicly.
