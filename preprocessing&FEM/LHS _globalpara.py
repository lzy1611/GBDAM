import numpy as np
import random
import pandas as pd

'''
Purpose of this file:
1. Generate a sample parameter matrix based on input variable ranges (numpy matrix) 
   and required sample numbers. Use ParameterArray function for execution.
'''


def Partition(number_of_sample, limit_array):
    """
    Partition variable ranges into intervals based on sample size
    :param number_of_sample: Required number of output samples
    :param limit_array: Variable range matrix of shape (m, 2),
                        m = number of variables, 2 = [lower, upper] bounds
    :return: Partitioned variable intervals as 3D matrix (layers = variables)
    """
    # Create coefficient matrices for interval partitioning
    coefficient_lower = np.zeros((number_of_sample, 2))
    coefficient_upper = np.zeros((number_of_sample, 2))

    # Calculate partitioning coefficients
    for i in range(number_of_sample):
        coefficient_lower[i, 0] = 1 - i / number_of_sample
        coefficient_lower[i, 1] = i / number_of_sample
        coefficient_upper[i, 0] = 1 - (i + 1) / number_of_sample
        coefficient_upper[i, 1] = (i + 1) / number_of_sample

    # Calculate partitioned ranges
    partition_lower = coefficient_lower @ limit_array.T  # Lower bounds matrix
    partition_upper = coefficient_upper @ limit_array.T  # Upper bounds matrix

    # Stack bounds to create 3D matrix (variables × samples × bounds)
    partition_range = np.dstack((partition_lower.T, partition_upper.T))
    return partition_range


def Representative(partition_range):
    """
    Generate random representative points within partitioned intervals
    :param partition_range: 3D matrix of shape (m, N, 2),
                             m = variables, N = samples, 2 = [lower, upper]
    :return: Matrix of representative points (N × m)
    """
    num_variables = partition_range.shape[0]
    num_samples = partition_range.shape[1]

    # Initialize random coefficient matrix
    coefficient_random = np.zeros((num_variables, num_samples, 2))
    representative_random = np.zeros((num_samples, num_variables))

    # Generate random coefficients for each variable and sample
    for m in range(num_variables):
        for i in range(num_samples):
            y = random.random()
            coefficient_random[m, i, 0] = 1 - y
            coefficient_random[m, i, 1] = y

    # Calculate representative points using element-wise multiplication
    temp_arr = partition_range * coefficient_random
    for j in range(num_variables):
        representative_random[:, j] = temp_arr[j, :, 0] + temp_arr[j, :, 1]
    return representative_random


def Rearrange(arr_random):
    """
    Shuffle values within each column of the matrix
    :param arr_random: Input matrix of shape (N, m)
    :return: Matrix with shuffled columns
    """
    for i in range(arr_random.shape[1]):
        np.random.shuffle(arr_random[:, i])
    return arr_random


def ParameterArray(limitArray, sampleNumber):
    """
    Generate LHS sample parameter matrix
    :param limitArray: Variable bounds matrix of shape (m, 2)
    :param sampleNumber: Required number of samples
    :return: Sample parameter matrix of shape (N, m)
    """
    arr = Partition(sampleNumber, limitArray)
    parametersMatrix = Rearrange(Representative(arr))
    return parametersMatrix


''' Class Definitions '''


class DoE(object):
    """
    Base class for Design of Experiments
    :param name_value: List of variable names
    :param bounds: Variable bounds matrix of shape (m, 2)
    """

    def __init__(self, name_value, bounds):
        self.name = name_value  # Variable names
        self.bounds = bounds  # Variable bounds [min, max]
        self.type = "DoE"  # Experiment type
        self.result = None  # Sampling results


class DoE_LHS(DoE):
    """ Latin Hypercube Sampling implementation """

    def __init__(self, name_value, bounds, N):
        """
        :param N: Number of samples to generate
        """
        super().__init__(name_value, bounds)
        self.type = "LHS"
        self.ParameterArray = ParameterArray(bounds, N)  # Generate samples
        self.N = N  # Number of samples

    def write_to_csv(self):
        """ Save generated samples to CSV file """
        sample_data = pd.DataFrame(self.ParameterArray, columns=self.name)
        sample_data.to_csv("LHS_globalparameters-test9.csv")


''' Usage Example '''

# Define variable bounds (diameter [mm], relative thickness [%])
arr_limit = np.array([[30, 0.035],
                      [70, 0.1]]).T

# Variable names
name_value = ["diameter", "relative thickness"]

# Generate 120 LHS samples
q = DoE_LHS(N=120, bounds=arr_limit, name_value=name_value)
q.write_to_csv()  # Write results to CSV