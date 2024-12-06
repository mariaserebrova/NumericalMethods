import numpy as np
from package.methods.newton_cotes import *
from package.methods.gauss import *
from package.methods.richardson import *



def f(x):
    return(3.5 * np.cos(1.5 * x) * np.exp(x / 4) + 4 * np.sin(3.5 * x) * np.exp(-1 * 3 * x) + 4 * x)
def quad_left_rect(func, a: float, b: float, num_partitions: int = 100) -> float:
    """Квадратурная формула левого прямоугольника. АСТ = 0"""
    result = 0

    for i in range(num_partitions):
        l_border = a + (b - a) * i / num_partitions
        r_border = a + (b - a) * (i + 1) / num_partitions

        result += (r_border - l_border) * func(l_border)

    return result


def quad_right_rect(func, a: float, b: float, num_partitions: int = 100) -> float:
    """Квадратурная формула правого прямоугольника. АСТ = 0"""
    result = 0

    for i in range(num_partitions):
        l_border = a + (b - a) * i / num_partitions
        r_border = a + (b - a) * (i + 1) / num_partitions

        result += (r_border - l_border) * func(r_border)

    return result


def quad_middle_rect(func, a: float, b: float, num_partitions: int = 100) -> float:
    """Квадратурная формула среднего прямоугольника. АСТ = 1"""
    result = 0

    for i in range(num_partitions):
        l_border = a + (b - a) * i / num_partitions
        r_border = a + (b - a) * (i + 1) / num_partitions

        result += (r_border - l_border) * func((l_border + r_border) / 2)

    return result


def quad_trapezia(func, a: float, b: float, num_partitions: int = 100) -> float:
    """Квадратурная формула трапеции. АСТ = 1"""
    result = 0

    for i in range(num_partitions):
        l_border = a + (b - a) * i / num_partitions
        r_border = a + (b - a) * (i + 1) / num_partitions

        result += (r_border - l_border) * (func(l_border) + func(r_border)) / 2

    return result


def quad_simpson(func, a: float, b: float, num_partitions: int = 100) -> float:
    """Квадратурная формула Симпсона. АСТ = 3"""
    result = 0

    for i in range(num_partitions):
        l_border = a + (b - a) * i / num_partitions
        r_border = a + (b - a) * (i + 1) / num_partitions

        result += (r_border - l_border) * (func(l_border) + 4 * func((l_border + r_border) / 2) + func(r_border)) / 6

    return result


def quad(func, a: float, b: float, method: str = 'Simpson',
         alpha: float = 0, beta: float = 0, num_partitions: int = 100) -> float:
    if method == 'Left Rectangle':
        return quad_left_rect(func, a, b, num_partitions)

    elif method == 'Right Rectangle':
        return quad_right_rect(func, a, b, num_partitions)

    elif method == 'Middle Rectangle':
        return quad_middle_rect(func, a, b, num_partitions)

    elif method == 'Trapezia':
        return quad_trapezia(func, a, b, num_partitions)

    elif method == 'Simpson':
        return quad_simpson(func, a, b, num_partitions)

    elif method == 'Gauss':
        return quad_gauss(func, a, b, alpha, beta, num_partitions)

    elif method == 'Newton Cotes':
        return quad_newton_cotes(func, a, b, alpha, beta, num_partitions)

# def left_rectangular(a, b, n):
#     h = (b - a) / n
#     res = 0

#     for i in range(n):
#         res += ((a + h * (i + 1)) - (a + h * i)) * f(a + h * i)

#     return res


# def middle_rectangular(a, b, n):
#     h = (b - a) / n
#     res = 0

#     for i in range(n):
#         mid_point = (a + h * i) + h / 2
#         res += f(mid_point) * h
#     return res


# def trapezoid(a, b, n):
#     h = (b - a) / n
#     res = 0

#     for i in range(n):
#         res += (h / 2) * (f(a + h * i) + f(a + h * (i + 1)))

#     return res


# def simpson(a, b, n):
#    h = (b - a) / (2 * n)

#    res = f(a) + f(b)

#    for i in range(1, 2 * n):
#        if i % 2 != 0:
#            res += 4 * f(a + i * h)
#        else:
#            res += 2 * f(a + i * h)

#    return res * h / 3
