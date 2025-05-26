import numpy as np
import random
import pandas as pd

'''
File Purpose:
1. Generate sample parameter matrix using Latin Hypercube Sampling (LHS)
2. Input: Variable range matrix (shape: (m,2)) and sample size
3. Output: Sample matrix (shape: (N,m))
Use ParameterArray function for core functionality
'''


def Partition(number_of_sample, limit_array):
    """
    Partition variable ranges into stratified intervals for LHS
    :param number_of_sample: Required sample size (N)
    :param limit_array: Variable bounds matrix (m,2), m = number of variables
    :return: 3D matrix (m x N x 2) of partitioned intervals
    """
    # Initialize coefficient matrices for interval calculation
    coefficient_lower = np.zeros((number_of_sample, 2))
    coefficient_upper = np.zeros((number_of_sample, 2))

    # Calculate partitioning coefficients
    for i in range(number_of_sample):
        coefficient_lower[i, 0] = 1 - i / number_of_sample
        coefficient_lower[i, 1] = i / number_of_sample
        coefficient_upper[i, 0] = 1 - (i + 1) / number_of_sample
        coefficient_upper[i, 1] = (i + 1) / number_of_sample

    # Calculate interval bounds using matrix multiplication
    partition_lower = coefficient_lower @ limit_array.T  # Lower bounds matrix
    partition_upper = coefficient_upper @ limit_array.T  # Upper bounds matrix

    # Combine into 3D matrix (variables × samples × bounds)
    partition_range = np.dstack((partition_lower.T, partition_upper.T))
    return partition_range


def Representative(partition_range):
    """
    Generate random representative points within stratified intervals
    :param partition_range: 3D interval matrix (m x N x 2)
    :return: Representative point matrix (N x m)
    """
    num_variables = partition_range.shape[0]
    num_samples = partition_range.shape[1]

    # Initialize random coefficient matrix
    coefficient_random = np.zeros((num_variables, num_samples, 2))
    representative_random = np.zeros((num_samples, num_variables))

    # Generate random coefficients for stratified sampling
    for m in range(num_variables):
        for i in range(num_samples):
            y = random.random()
            coefficient_random[m, i, 0] = 1 - y
            coefficient_random[m, i, 1] = y

    # Calculate representative points using element-wise operations
    temp_arr = partition_range * coefficient_random
    for j in range(num_variables):
        representative_random[:, j] = temp_arr[j, :, 0] + temp_arr[j, :, 1]
    return representative_random


def Rearrange(arr_random):
    """
    Shuffle values within each column to ensure statistical independence
    :param arr_random: Input matrix (N x m)
    :return: Column-wise shuffled matrix
    """
    for i in range(arr_random.shape[1]):
        np.random.shuffle(arr_random[:, i])
    return arr_random


def ParameterArray(limitArray, sampleNumber):
    """
    Main LHS sampling function
    :param limitArray: Variable bounds matrix (m x 2)
    :param sampleNumber: Required sample size (N)
    :return: LHS sample matrix (N x m)
    """
    arr = Partition(sampleNumber, limitArray)
    parametersMatrix = Rearrange(Representative(arr))
    return parametersMatrix


''' Class Implementations '''


class DoE:
    """
    Base class for Design of Experiments
    :param name_value: List of variable names
    :param bounds: Variable bounds matrix (m x 2)
    """

    def __init__(self, name_value, bounds):
        self.name = name_value  # Variable names
        self.bounds = bounds  # [[min1, max1], [min2, max2], ...]
        self.type = "DoE"  # Experiment type identifier
        self.result = None  # Sampling results storage


class DoE_LHS(DoE):
    """ Latin Hypercube Sampling implementation """

    def __init__(self, name_value, bounds, N):
        """
        :param N: Number of LHS samples to generate
        """
        super().__init__(name_value, bounds)
        self.type = "LHS"
        self.ParameterArray = ParameterArray(bounds, N)  # Generated samples
        self.N = N  # Number of samples

    def write_to_csv(self):
        """ Save generated samples to CSV file """
        sample_data = pd.DataFrame(self.ParameterArray, columns=self.name)
        sample_data.to_csv("LHS_localparameters-test4-5.csv")


''' Implementation Example '''

# Define parameter ranges (min, max) for 16 variables
arr_limit = np.array([
    [150, 0, 30, 2.5, 0.5, 0.8, 2, 0.2, 1.2, 0, 0, 0, 0.1, 0.1, 0.1, 0.1],
    [300, 360, 135, 5, 1.5, 1.5, 5, 0.4, 1.6, 0.03, 0.03, 0.03, 0.4, 0.4, 0.4, 0.4]
]).T

# Variable names for 16 parameters
name_value = [
    "Y", "B", "C bending angle", "relative radius",
    "velocity of bending", "velocity of pressure",
    "num of balls", "relative ball thickness",
    "relative gap of balls", "gap of bending",
    "gap of pressure", "gap of wiper",
    "FTB", "FTP", "FTW", "FTM"
]

# Generate 120 LHS samples
q = DoE_LHS(N=120, bounds=arr_limit, name_value=name_value)
q.write_to_csv()  # Export to CSV