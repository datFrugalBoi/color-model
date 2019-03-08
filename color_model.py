from __future__ import print_function
from __future__ import print_function
from __future__ import division

# author:  Cole Brendel 
# created on: 2019-02-26

# a multiclass logisitic regressor to predict ratings for color schemes

'''
This model assumes that ratings (1 - 5) are ordinal categories.

The model predicts a rating `class` given 6 values for R,G,B,H,S,L values.

'''

# for debugging
import pdb

# basic imports
import argparse
from datetime import datetime
import time

# dataframes and data prep
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# model
from sklearn.linear_model import LogisticRegression

# model metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def write_results(content,filename):
    '''
    A simple function writing arbitrary content to a txt file.

    :rtype: object
    :param content: the data to be written
    :param filename: name of file to write content to
    :return results: a txt file
    '''
    results = open(filename, "a") #create a new file or append to existing
    results.write(content + '\n')
    return results


def read_df(dataset):
    '''
    Takes csv files and converts to pandas dataframe.

    :param dataset: the input csv
    :return df: a pandas dataframe
    '''
    print('Reading in data.')
    t0 = time.time()
    try:
        df = pd.read_csv(dataset, low_memory=False)

        row_count = df.shape[0]
        col_count = df.shape[1]

        print('Read in dataset: {0} rows, {1} columns.'.format(str(row_count), str(col_count)))

        print ('\n Dataframe contains the following cols: \n')

        # inspect the cols and dtypes
        print (df.info())

        t1 = time.time()
        total_n = t1 - t0
        print ("Total time reading in data: {0}".format(str(total_n)))

        return df
    except Exception as e:
        print (e)
        print ('Error reading data!')


# build df for the logistic regression model
def build_model_df(df):
    '''
    Takes the dataframe from read_df() and prepares for modeling.

    - applies StandardScaler to normalize data

    :param df: input training data
    :return df_mdl:  dfsready for modelling.
    '''
    print ('Building dfs for models.')
    t0 = time.time()

    # 'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
    # 'S1', 'S2', 'S3', 'S4', 'S5', 'S6',
    # 'L1', 'L2', 'L3', 'L4', 'L5', 'L6',

    # This feature set is for training the classifier. 
    model_cols = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 
                  'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 
                  'B1', 'B2', 'B3', 'B4', 'B5', 'B6','rating']  # last one is the dependent var

    # pare dfs to model_cols
    df_mdl = df[model_cols].copy()

    # rename rating to target variable
    df_mdl.rename(columns={'rating': 'target_variable'}, inplace=True)

    # Get column names first to run the scaler. !! DO NOT SCALE TARGET_VARIABLE !! 
    col_names = df_mdl.columns.values.tolist()
    to_scale = [c for c in col_names if c != 'target_variable']

    # convert stray inf, -inf to np.nan
    df_mdl.replace([np.inf, -np.inf], np.nan, inplace=True)

    # nuke missing values
    df_mdl.dropna(axis=0, # rows 
              how='any', # if any are np.nan, destroy
              inplace=True) # do not copy df, modify in place

    # Create the Scaler object
    scaler = preprocessing.StandardScaler()

    # Impute missing values
    imp = SimpleImputer(strategy="mean")

    # fit the Scaler to the dfs
    df_mdl[to_scale] = scaler.fit_transform(imp.fit_transform(df_mdl[to_scale].copy()))
    df_mdl = pd.DataFrame(df_mdl, columns=col_names)

    t1 = time.time()
    total_n = t1 - t0
    print ("Total time building df for modelling: {0}".format(str(total_n)))

    df_mdl['target_variable'] = df_mdl['target_variable'].astype(int)
    return df_mdl


def train_test(df):
    '''
    Split data into training and test sets. 

    :param df: input training data
    :return df, X_train, X_test, y_train, y_test: train, test splits for model building.
    '''
    print ('Splitting data into training and test sets.')
    t0 = time.time()

    # prepare predictors and target
    X = df.loc[:, df.columns != 'target_variable']
    y = df.loc[:, df.columns == 'target_variable']

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

        t1 = time.time()
        total_n = t1 - t0
        print ("Total time splitting data: {0}".format(str(total_n)))

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print (e)
        print ('Something went horribly wrong!')
        return False


def build_a_model(X_train, y_train):
    '''
    Build a logistic regression model to predict color ratings.

    :param X_train: features to learn
    :param y_train: targets to predict
    :return best_model: a trained logistic regressor 
    '''
    print ('Building a layer.')
    t0 = time.time()
    # initialize a model
    logreg = LogisticRegression(random_state=8675309,
                                class_weight='balanced',
                                penalty='l2', # error-term in loss function
                                C=1291, # regularization
                                solver="newton-cg", # use newton's method to solve gradient descent
                                multi_class='multinomial', #multiclass
                                warm_start=True, # successive CV fits use coeffs from last model
                                n_jobs=-1) # utilize all available cpu cores to speed-up training

    best_model = logreg.fit(X_train, y_train.values.ravel()) #p3.8xl may be useful here

    t1 = time.time()
    total_n = t1 - t0
    print ("Total time building this model: {0}".format(str(total_n)))

    return best_model


def model_eval(best_model, X, y, y_pred):
    '''
    Convienence functions to montior model performance. 

    :param best_model: trained model object
    :param X: features learned from
    :param y: true targets
    :param y_pred: targets predicted
    :return matrix, report: reports and metrics on classification performance
    '''

    try:
        print('Accuracy of classifier on test set: {:.2f}'.format(best_model.score(X, y)))
    except:
        pass

    matrix = str(confusion_matrix(y, y_pred))
    #print(matrix)
    targets = ['0','1','2','3','4']
    #target_names=targets
    report = classification_report(y, y_pred, digits=5)
    print(report)

    return matrix, report

# Time to run validations.
def run_validation(best_model, X_test, y_test):
    '''

    Validates the logistic regression model on our test set.

    :param X_test: Traing features
    :param y_test: Training targets
    :return:
    '''

    print ("Running validations.")
    t0 = time.time()

    # Predict target vector
    print ("Predicting categories on validation set.")
    t_0 = time.time()
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    t_1 = time.time()
    total_n_ = t_1 - t_0
    print ("Total time running model on validation set: {0}".format(str(total_n_)))

    # evaluate the model
    matrix, report = model_eval(best_model, X_test, y_test, y_pred)

    t1 = time.time()
    total_n = t1 - t0
    print ("Total time running validation: {0}".format(str(total_n)))
    return matrix, report


def pipeline(df, results):
    '''
    Orchestrates the entire modelling pipeline.

    :param df: input training data
    :param validation_set: input validation data
    :param goldens: golden rcof data from new AMDS annotations
    :param results: file to write results
    :return:
    '''

    print ('Begin modelling pipeline.')
    t0 = time.time()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

    # 1. read-in data
    df = read_df(df)

    # 2. build dfs for modelling
    df = build_model_df(df)

    # 3. train-test split
    X_train, X_test, y_train, y_test = train_test(df)

    # 5. Build model
    best_model = build_a_model(X_train, y_train)

    # 6. Run validations
    matrix, report = run_validation(best_model, X_test, y_test) 

    ##recording
    write_results('Validation Results', results)
    write_results(str(datetime.now()), results)
    write_results(str(matrix), results)
    write_results(str(report), results)
    print (str(matrix))
    print (str(report))

    t1 = time.time()
    total_time = t1 - t0
    print ('Modelling pipeline complete.')
    print ("Total time to run through pipeline : {0}".format(str(total_time)))

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################


# main function
def main():

    # get arguments
    parser = argparse.ArgumentParser(
        description='Script to build a logistic regression model for color scheme ratings.',
        epilog='Example: python color_model.py -df ./input_data -o ./results' )
    parser.add_argument('--dataframe', '-df', required=True, help='input file')
    parser.add_argument('--output_file', '-o', required=True, help='(path) filename of output file')
    args = parser.parse_args()

    # kick-off modelling

    pipeline(args.dataframe, args.output_file)


if __name__ == '__main__':
    main()