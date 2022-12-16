import numpy as np
import pandas as pd
from sklearn import metrics

def train_test_val(*arrays, val_size, test_size, verbose=True):
    '''
    Split arrays into training, validation and testing arrays.
    
    ---------
    *arrays : array or dataframe
        Arbitrary number of arrays to split.
    val_size : int or float
        Size of validation data in absolute or relative terms.
    test_ size : int or float
        Size of testing data in absolute or relative terms.
    verbose : bool, default True
        Whether to print array lengths to the console.
    
    Returns
    -------
    train, val, test : list of arrays
        Training, validation and testing arrays split using val_size and test_size.
    '''
    array_len = len(arrays[0])

    if not all(len(array) == array_len for array in arrays):
        raise ValueError('Arrays must be of equal length!')

    if type(val_size) is float:
        val_size = int(array_len * val_size)
    
    if type(test_size) is float:
        test_size = int(array_len * test_size)
    
    train_size = array_len - (val_size + test_size)

    if verbose == True:
        print(f'{"Array":<10}{"Length (n)":>12}{"Length (%)":>12}')
        print(f'{34 * "-"}')
        print(f'{"training":<10}{train_size:>12}{(train_size / array_len):12.2%}')
        print(f'{"validation":<10}{val_size:>12}{(val_size / array_len):12.2%}')
        print(f'{"testing":<10}{test_size:>12}{(test_size / array_len):12.2%}')
        print(f'{34 * "-"}')
        print(f'{"total":<10}{array_len:>12}{(array_len / array_len):12.2%}')

    array_list = []

    for array in arrays:
        train = array[:train_size]
        val = array[train_size:-test_size]
        test = array[-test_size:]

        array_list.extend([train, val, test])

    return array_list

def to_sequences(data, length=1):
    '''
    Transform data into three dimensional sequence-based lstm input array.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe where the first column is the output array.
    length : int > 0, default 1
        Number of timesteps to use for sequences.

    Returns
    -------
    X : array
        Three dimensional lstm input array with data points turned into sequences.
    '''
    n_samples = data.shape[0] - (length - 1)# subtracting sequence length - 1 from array length to ensure all samples have enough data points
    n_features = data.shape[1]

    X = np.zeros((n_samples, length, n_features))

    for i in range(n_samples):
        sequence = data.values[i:i + length, :]
        X[i, :, :] = sequence.reshape(1, length, n_features)
    
    return X

def calc_metrics(y_true, y_pred, verbose=True, save_as=False):
    '''
    Calculate performance metrics for machine learning model.
    
    Parameters
    ----------
    y_true : array
        Actual values.
    y_pred : array
        Forecasted values.
    verbose : bool, default True
        Whether to print metrics to the console.
    
    Returns
    -------
    perf_metrics : dict
        Calculated performance metrics.
    '''
    cpe = sum(abs(y_true - y_pred)) / sum(y_true)
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)# (sum((y_true - y_pred) ** 2) / len(y_true)) ** 0.5
    mae = metrics.mean_absolute_error(y_true, y_pred)# sum(abs(y_true - y_pred)) / len(y_true)
    r2 = metrics.r2_score(y_true, y_pred)# 1 - (sum((y_true - y_pred) ** 2) / sum((y_true - np.mean(y_true)) ** 2))
    # Since actuals are close to 0, MAPE would result in a very high value and is therefore not used

    perf_metrics = {
        'CPE': cpe,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
        }
    
    if verbose == True:
        print(f'{"Metric" : <14}{"Value" : ^7}')
        print(f'---------------------')
        for perf_metric in perf_metrics:
            print(f'{perf_metric : <14}{round(perf_metrics[perf_metric], 5) : >7}')
    
    if save_as != False:
        pd.DataFrame(perf_metrics, index=[0]).to_csv(save_as, index=False)

    return perf_metrics