# TO DO:
# Add more methods for train/test split

def train_test_split_by_month(df, test_months):
    '''
    Split data into training and test datasets based off the month
    
    Inputs:
    df (dataframe) - dataframe containing the training and test data needed to be split
    test_months (list) - months to be used as test data, must be len = 2 strings

    Outputs:
    train_df, test_df (dataframe)- dataframes divided based off the month
    '''

    # Add month to the df if it is not there
    if not ('month' in (df.columns)):
        df['month'] = df.index.get_level_values('time').month

    train_df = df[~df['month'].isin(test_months)]
    test_df  = df[df['month'].isin(test_months)]

    return train_df, test_df

def set_X_y(X, y, df):
    '''
    Create datasets for the X (i.e. predictors) variables and the y (i.e. target) variable.

    Inputs:
    X (list) - variables to be used as the X variable for the model
    Y (str) - variable to be used as the y variable for the model
    df (dataframe) - dataset to read the X and y variables from

    Outputs:
    X_df (dataframe) - dataset containing the X variables
    y_df (dataframe) - dataset containing the y variable
    '''
    
    X_df = df[X]
    y_df = df[[y]]
    return X_df, y_df


def prepare_data(df, X, y, test_months=None):
    '''
    Prepare all the datasets needed to train and test a machine learning model.

    Inputs:
    df (dataframe) - dataset containing all the data to draw from
    X (list) - variables to be used as the X variable for the model
    Y (str) - variable to be used as the y variable for the model
    test_months (list) - months to be used as test data, must be len = 2 strings

    Outputs:
    X_train, X_test, y_train, y_test
    '''

    if test_months:
        train_df, test_df = train_test_split_by_month(df, test_months)
    
        X_train, y_train = set_X_y(X, y, train_df)
        X_test, y_test = set_X_y(X, y, test_df)
    
        return X_train, X_test, y_train, y_test
    else:
        X, y = set_X_y(X, y, df)
        return X, y
    
    