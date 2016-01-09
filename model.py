# All imports
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import time


# The Model Builder class to buils and train the Logistic Regression based model
class ModelBuilder:
    # Function to drop unwanted columns
    def drop_unwanted_columns(self, df, cols_to_be_dropped):
        for col in cols_to_be_dropped:
            df.drop(col, 1, inplace=True)
        return df

    # Function to transform variables into their dummy equivalent for fields having 50-80% NAs
    def dummify_unfeasible_columns(self, df, cols_to_be_dummified):
        for col in cols_to_be_dummified:
            df[col] = df[col].notnull().astype('int64')
        return df

    # Function to impute variables with less than 10% NAs with the mean of the not np.nan values in the field
    def impute_means(self, df, cols_NOT_to_impute_with_means):
        for col in df.columns:
            if col not in cols_NOT_to_impute_with_means:
                df[col].fillna(value=df[col].mean(), inplace=True)
        return df

    # Function to impute variables with their most collinear variables for variable with 10-50% NAs
    def impute_ally(self, df, cols_to_impute_with_allies, cols_allies):
        for col1, col2 in zip(cols_to_impute_with_allies, cols_allies):
            ds = df[[col1, col2]]   # Extracting the two columns from the dataframe
            ds.dropna(how='any', inplace=True)  # Dropping all the NAs to train the Linear Model
            regr = linear_model.LinearRegression(fit_intercept=True)    # Instantiating an object of the Lin Regr Model
            x = ds[col2].reshape(len(ds), 1)    # Reshaping the columns
            y = ds[col1].reshape(len(ds), 1)
            regr.fit(x, y)  # Fitting the linear regression model
            # Predicting the missing columns values from the allied column values via the trained Lin Regr Model
            prediction = regr.predict(df[col2].reshape(len(df), 1))
            pred_series = pd.Series(prediction.ravel(), index=(df['ID']-1))
            df[col1] = df[col1].fillna(pred_series)
        return df

    # Function to handle all the NAs and return back a data frame clean of NAs
    def clean_data(self, df):
        # Remove those entries from the data set for which Bureau Variables are not available
        df = df[pd.notnull(df['B_CURRENT_INQ_NBR']) & pd.notnull(df['B_6_mth_INQ_NBR'])]
        # List of columns to be dropped because they don't make semantic sense
        cols_to_be_dropped_no_sense = ['AS_OF', 'EOMCYCDEL', 'CHARGEOFF', 'DELQ_FLAG']
        # List of columns to be dropped because they show very high correlation with some other variables
        cols_to_be_dropped_high_corr = ['INT_CASH_MTD', 'INT_CASH_PC']
        # List of columns to be dropped because they have more than 80% NAs
        cols_to_be_dropped_nas = ['3_MTH_SPEND_VS_CASH_ADCV', '3MTH_ABD_VS_3MTH_CASH_ADVC', \
                                  '6_MTH_SPEND_VS_CASH_ADCV', '6MTH_ABD_VS_6MTH_CASH_ADVC']
        cols_to_category = ['Target Flag', 'DEROG_FLAG', 'CLI', 'INQ_INCREASE_6MTH']
        cols_to_be_dummified = ['B_6_mth_FIN_RVOLV_ALL_TRD_AVG_BAL', 'B_CURRENT_FIN_RVOLV_ALL_TRD_AVG_BAL', \
                                    'B_6_mth_MTH_LST_BNKCRD_TRD_DELQ', 'B_6_mth_MTGE_TRD_BAL_TO_CR_LMT_RAT', \
                                    'B_CURRENT_MTGE_TRD_BAL_TO_CR_LMT_RAT', 'B_CURRENT_MTH_LST_BNKCRD_TRD_DELQ']
        cols_to_impute_with_allies = ['B_CURRENT_RTL_TRD_TOT_BAL_CR_LMT_RAT', 'B_6_mth_ISTLMT_TRD_BAL_TO_CR_LMT_RAT', \
                                      'B_CURRENT_ISTLMT_TRD_BAL_TO_CR_LMT_RAT', 'B_6_mth_RTL_TRD_TOT_BAL_CR_LMT_RAT', \
                                      'ADB_VS_3MTH_SPEND', 'B_6_mth_MTH_SINCE_MOST_RCNT_DELQ', 'B_6_mth_TRD_HG_DELQ_NBR', \
                                      'B_CURRENT_MTH_SINCE_MOST_RCNT_DELQ', 'B_CURRENT_TRD_HG_DELQ_NBR', \
                                      'ADB_VS_6MTH_SPEND', 'B_CURRENT_MTH_SINCE_MOST_RCNT_INQ', \
                                      'B_6_mth_MTH_SINCE_MOST_RCNT_INQ']
        cols_allies = ['B_CURRENT_RVLV_TRD_TOT_BAL_TO_CR_LMT_RAT', 'B_6_mth_TOT_BAL_TO_CR_LMT_RAT', \
                       'B_CURRENT_TOT_BAL_TO_CR_LMT_RAT', 'B_6_mth_RVLV_TRD_TOT_BAL_TO_CR_LMT_RAT', 'AVG_3M_BAL', \
                       'SCORE_FICO', 'B_6_mth_RTG_90_DY_NBR', 'SCORE_FICO', 'B_6_mth_RTG_60_DY_NBR', 'AVG_6M_BAL', \
                       'B_CURRENT_MTH_SINCE_MOST_RCNT_TRD_OPN', 'B_CURRENT_MTH_SINCE_MOST_RCNT_TRD_OPN']
        cols_to_be_dropped = cols_to_be_dropped_no_sense + cols_to_be_dropped_high_corr + cols_to_be_dropped_nas
        df = self.drop_unwanted_columns(df, cols_to_be_dropped)
        df = self.dummify_unfeasible_columns(df, cols_to_be_dummified)
        cols_NOT_to_impute_with_means = cols_to_be_dropped + cols_to_be_dummified + \
                                        cols_to_impute_with_allies + cols_to_category
        df = self.impute_means(df, cols_NOT_to_impute_with_means)
        df = self.impute_ally(df, cols_to_impute_with_allies, cols_allies)
        return df

    # Function to build the Logistic Regression base model
    def build_model(self, df):
        # Rearranging the columns a bit
        id = pd.DataFrame(df.pop('ID'))
        target = pd.DataFrame(df.pop('Target Flag'))
        id = id.join(target)
        df = id.join(df)
        y = df['Target Flag']
        training_cols = ['3MTH_PAY_VS_3MTH_ADB', 'ADB', 'B_6_mth_RTG_30_DY_NBR','B_6_mth_TRD_CURR_PAST_DUE_NBR',\
                         'B_6_mth_TRD_NEVER_DELQ_PCT', 'B_CURRENT_BNKCRD_TRD_GT_75_LMT_PCT',\
                         'B_CURRENT_FIN_RVOLV_ALL_TRD_AVG_BAL', 'B_CURRENT_INQ_LST_6_MTH_NBR',\
                         'B_CURRENT_MTH_SINCE_MOST_RCNT_DELQ', 'B_CURRENT_TOT_BAL_TO_CR_LMT_RAT',\
                         'B_CURRENT_TRD_HG_DELQ_NBR', 'SCORE_FICO', 'SPEND_3MTH/ADB_3MTH', 'UTILIZATION_RATE']
        for col in df.columns.tolist():
            if col not in training_cols:
                df = df.drop(col, 1)    # Keeping only the training columns in the data frame
        x = df
        # Splitting the data set into train-test ratio of 70:30
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        logit = sm.Logit(y_train, x_train)  # Instantiating a Logistic Regression Model Object
        model = logit.fit_regularized()   # Fitting the training set in the model

        f = open("LogRegSummary.txt", "w")
        f.write(str(model.summary2()))    # Writing the regression summary in a file
        f.close()
        print(model.summary2())   # Printing the regression summary
        return model

    def __init__(self):
        t = time.time()
        print("Reading the data file..."),
        df = pd.read_excel("creditcards.xlsx")
        print("Success [" + str(time.time()-t) + " seconds]")
        t = time.time()
        print("Cleaning data..."),
        df = self.clean_data(df)
        print("Data Cleaned [" + str(time.time()-t) + " seconds]")
        t = time.time()
        print("Building Model..."),
        model = self.build_model(df)
        print("Model Trained [" + str(time.time()-t) + " seconds]")


# The main method
def main():
    start_time = time.time()
    ModelBuilder()
    print("Total Time Taken:", time.time()-start_time, "seconds")

# Invocation of the main method
if __name__ == '__main__':
    main()