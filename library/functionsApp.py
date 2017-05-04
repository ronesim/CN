import math
import os
import random


def horner(coeff, x_value):
    result = coeff[0]
    for i in range(1, len(coeff)):
        result = result * x_value + coeff[i]
    return result


def derivative(coeff):
    derivative_coeff = []
    n = len(coeff) - 1
    for i in range(0, len(coeff) - 1):
        derivative_coeff.append(coeff[i] * n)
        n -= 1
    return derivative_coeff


def get_values_interval(coeff):
    R = (coeff[0] + max(abs(x) for x in coeff[1:])) / coeff[0]
    return (-R, R)


def sgn(value):
    if value < 0:
        return -1
    return 1


def find_root(coeff, x_value, eps):
    first_derivative = derivative(coeff)
    second_derivative = derivative(first_derivative)
    N = len(coeff)
    x_k = x_value
    delta_x = 0
    for step in range(0, 10000):
        y_value = horner(coeff, x_k)
        y_first_value = horner(first_derivative, x_k)
        y_second_value = horner(second_derivative, x_k)

        H = ((N - 1) ** 2) * (y_first_value ** 2) - N * (N - 1) * y_value * y_second_value
        if (H < 0):
            return "Divergent"

        denominator = y_first_value + sgn(y_first_value) * math.sqrt(H)
        if denominator >= -eps and denominator <= eps:
            return "Divergent"
        delta_x = (N * y_value) / denominator

        if abs(delta_x) <= eps:
            x_k = x_k - delta_x
            break

        if abs(delta_x) >= 10 ** 8:
            return "Divergent"

        x_k = x_k - delta_x

    if abs(delta_x) <= eps:
        return x_k

    return "Divergent"


def find_minimum(coeff, x_value1, x_value2, eps):
    first_derivative = derivative(coeff)
    x_k1 = x_value1
    x_k2 = x_value2
    delta_x = 0
    for step in range(0, 10000):
        y_first_value_1 = horner(first_derivative, x_k1)
        y_first_value_2 = horner(first_derivative, x_k2)

        denominator = y_first_value_2 - y_first_value_1
        if denominator >= -eps and denominator <= eps:
            delta_x = 10 ** (-5)
        else:
            delta_x = ((x_k2 - x_k1) * y_first_value_2) / denominator

        x_k1 = x_k2 - delta_x

        if abs(delta_x) <= eps:
            break

        if abs(delta_x) >= 10 ** 8:
            return "Divergent"

    if abs(delta_x) <= eps:
        return x_k1

    return "Divergent"


def main_function(coeff):
    result = []
    interval = get_values_interval(coeff)
    for i in range(0, 50):
        x_value = random.randint(int(interval[0]), int(interval[1]))
        root = find_root(coeff, x_value, 10 ** (-16))
        if type(root) == str and root is "Divergent":
            continue
        result.append(root)

    result = list(set(result))

    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    APP_UPLOADS = os.path.join(APP_ROOT, 'example/roots.txt')
    file = open(APP_UPLOADS, 'w')
    file.write(", ".join(repr(e) for e in result))

    # minimum of a function
    result_minim = []
    interval_min = get_values_interval(derivative(coeff))
    for i in range(0, 50):
        x_value_1 = random.randint(int(interval_min[0]), int(interval_min[1]))
        x_value_2 = random.randint(int(interval_min[0]), int(interval_min[1]))
        min_value = find_minimum(coeff, x_value_1, x_value_2, 10 ** (-16))
        if type(min_value) == str and min_value is "Divergent":
            continue
        result_minim.append(min_value)

    result_minim = list(set(result_minim))
    return sorted(result), sorted(result_minim)
