import math

import matplotlib.pyplot as plt
import numpy as np


def generate_random_points(n, x0, xn, eps):
    points = np.random.uniform(x0 + eps, xn - eps, 2 * n)
    while (len(np.unique(points[:n])) < n):
        points = np.random.uniform(x0, xn, 2 * n)
    return np.unique(points[:n])


def get_numeric_function_values(points, my_function):
    ans = np.zeros(len(points))
    for index in range(0, len(points)):
        value = 1
        for coeff in range(0, len(my_function)):
            ans[index] = ans[index] + value * my_function[coeff]
            value = value * points[index]
    return ans


def aitken_method(x, y):
    """ 
    :param x: the array of abscissa
    :param y: the array of ordinates 
    :return:  the top elements of the Divided Difference Table used when 
             calculating Newton's Divided Differences
    """
    dividedDifferences = np.copy(y)
    for i in range(1, len(x)):
        for j in range(len(x) - 1, 0, -1):
            if j - i >= 0:
                dividedDifferences[j] = (dividedDifferences[j] - dividedDifferences[j - 1]) / (x[j] - x[j - i])
    return dividedDifferences


def numerical_interpolation(points, values, x_value):
    y = aitken_method(points, values)
    ans = y[0]
    for i in range(1, len(points)):
        aux_value = 1
        for j in range(0, i):
            aux_value = aux_value * (x_value - points[j])
        ans += aux_value * y[i]
    return ans


def trig_function(type, x):
    if type == 2:
        return math.sin(x) - math.cos(x)
    if type == 3:
        return math.sin(2 * x) + math.sin(x) + math.cos(3 * x)
    if type == 4:
        return math.sin(x) ** 2 - math.cos(x) ** 2


def get_trigonometric_function_values(type, points):
    ans = np.zeros(len(points))
    for i in range(0, len(points)):
        ans[i] = trig_function(type, points[i])
    return ans


def compute_matrix_T(points):
    T = np.zeros((len(points), len(points)))
    for row in range(0, len(points)):
        for col in range(0, len(points)):
            if col == 0:
                T[row, col] = 1
                continue
            if col % 2 == 1:
                T[row, col] = math.sin(((col + 1) / 2) * points[row])
                continue
            if col % 2 == 0:
                T[row, col] = math.cos((col / 2) * points[row])
    return T


def trigonometric_interpolation(x, x_value):
    ans = x[0]
    for i in range(1, len(x)):
        if i % 2 == 1:
            ans += x[i] * math.sin(x_value * ((i + 1) / 2))
        else:
            ans += x[i] * math.cos(x_value * (i / 2))
    return ans


def main_function(type, number_of_values, start, end, x_value):
    eps = 10 ** (-10)

    if type == 1:
        # define numeric function - coefficients
        my_function = [12, 0, 30, -12, 1]
        # my_function = [1, 0, 1]

        # generate random points and get the values for my function in that points
        points = generate_random_points(number_of_values - 1, start, end, eps)
        points = np.insert(points, 0, start)
        points = np.append(points, end)
        # points = [0 , 2, 4]
        values = get_numeric_function_values(points, my_function)

        aprox_value_interpolation = numerical_interpolation(points, values, x_value)
        real_value = get_numeric_function_values([x_value], my_function)
        print("Lagrange interpolation: ", aprox_value_interpolation)
        print("Real value: ", real_value[0])
        print("Difference: ", abs(aprox_value_interpolation - real_value[0]))

        points_to_plot = np.arange(0.0, 10.0, 1.0)
        values_to_plot = get_numeric_function_values(points_to_plot, my_function)
        interpolation_points = [numerical_interpolation(points_to_plot, values_to_plot, x) for x in points_to_plot]

        plt.plot(points_to_plot, values_to_plot, 'r', points_to_plot, interpolation_points, 'b--')
        plt.axis([0, 10, -200, 100])
        plt.show()

    else:
        # trigonometric function
        PI = math.pi
        points = generate_random_points(number_of_values, 0, end, eps)
        points = np.insert(points, 0, start)
        points = np.append(points, end)

        values = get_trigonometric_function_values(type, points)
        T = compute_matrix_T(points)
        values = np.reshape(values, (len(points), 1))

        # solve system TX = Y
        x = np.linalg.solve(T, values)

        aprox_value_interpolation = trigonometric_interpolation(x, x_value)
        real_value = trig_function(type, x_value)
        print("Trigonometric Interpolation: ", aprox_value_interpolation)
        print("Real value: ", real_value)
        print("Difference: ", abs(aprox_value_interpolation - real_value))

        points_to_plot = np.arange(start, end, 0.1)
        values_to_plot = get_trigonometric_function_values(type, points_to_plot)
        interpolation_result = []
        for point in points_to_plot:
            interpolation_result.append(trig_function(type, point))
        plt.plot(points_to_plot, values_to_plot, 'r', points_to_plot, interpolation_result, 'b--')
        plt.axis([-10, 10, -2, 2])
        plt.show()


main_function(1, 10, 1, 19, 3)
print("==================================")
PI = math.pi
main_function(2, 11, 0, 31 * PI / 16, 1.5)
