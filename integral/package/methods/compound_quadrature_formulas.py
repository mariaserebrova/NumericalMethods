import numpy as np


def f(x):
    return(3.5 * np.cos(1.5 * x) * np.exp(x / 4) + 4 * np.sin(3.5 * x) * np.exp(-1 * 3 * x) + 4 * x)


def left_rectangular(a, b, n):
    h = (b - a) / n
    res = 0

    for i in range(n):
        res += ((a + h * (i + 1)) - (a + h * i)) * f(a + h * i)

    return res


def middle_rectangular(a, b, n):
    h = (b - a) / n
    res = 0

    for i in range(n):
        mid_point = (a + h * i) + h / 2
        res += f(mid_point) * h
    return res


def trapezoid(a, b, n):
    h = (b - a) / n
    res = 0

    for i in range(n):
        res += (h / 2) * (f(a + h * i) + f(a + h * (i + 1)))

    return res


def simpson(a, b, n):
   h = (b - a) / (2 * n)

   res = f(a) + f(b)

   for i in range(1, 2 * n):
       if i % 2 != 0:
           res += 4 * f(a + i * h)
       else:
           res += 2 * f(a + i * h)

   return res * h / 3
