from math import *
import matplotlib.pyplot as plt
import numpy as np

measurements = [5, 6, 7, 9, 10]
motions = [1, 1, 2, 1, 1]

mean = 0
variance = 10000

measurement_uncertainty = 4
motion_uncertainty = 2


def guassian_function(mean, variance_square, x):
    '''
    :param mean: The mean of the distribution
    :param variance_square: The variance of the distribution squared
    :param x: Value to be measured
    :return: gaussian value
    '''

    coefficient = 1.0 / sqrt(2.0 * pi * variance_square)
    exponential = exp(-0.5 * (x - mean) ** 2 / variance_square)

    return coefficient * exponential


def update(mean1, var1, mean2, var2):
    '''
    Update gaussian parameters
    :param mean1: mean of first distribution
    :param var1: variance of first distribution
    :param mean2: mean of second distribution
    :param var2: variance of second distribution
    :return: new parameters
    '''

    new_mean = (var2 * mean1 + var1 * mean2) / (var2 + var1)
    new_var = 1 / (1 / var2 + 1 / var1)

    return [new_mean, new_var]


def predict(mean1, var1, mean2, var2):
    '''
    Return updated parameters after motion
    :param mean1: mean of first distribution
    :param var1: variance of first distribution
    :param mean2: mean of second distribution
    :param var2: variance of second distribution
    :return: new parameters after motion
    '''

    new_mean = mean1 + mean2
    new_var = var1 + var2

    return [new_mean, new_var]

for n in range(len(measurements)):
    mean, variance = update(mean, variance, measurements[n], measurement_uncertainty)
    print('Update: [{}, {}]'.format(mean, variance))

    mean, variance = predict(mean, variance, motions[n], motion_uncertainty)
    print('Predict: [{}, {}]'.format(mean, variance))

    # define a range of x values
    x_axis = np.arange(-20, 20, 0.1)

    # create a corresponding list of gaussian values
    g = []
    for x in x_axis:
        g.append(guassian_function(mean, variance, x))

    # plot the result
    plt.plot(x_axis, g)
    plt.show()