import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from package.methods.compound_quadrature_formulas import *
from package.methods.newton_cotes import *
from package.methods.gauss import *

def f(x):
    """Функция для интегрирования."""
    return (3.5 * np.cos(1.5 * x) * np.exp(x / 4) +
            4 * np.sin(3.5 * x) * np.exp(-3 * x) +
            4 * x)

def p(x, a=2.5, b=3.3, alpha=2/3, beta=0):
    """Весовая функция."""
    if x <= a or x >= b:
        return 0  # Возвращаем 0, если x выходит за пределы
    return (x - a) ** (-alpha) * (b - x) ** (-beta)

def F(x):
    """Комбинированная функция для интегрирования."""
    return f(x) * p(x)

def calculate_exact_integral(a, b):
    """Вычисляет точное значение интеграла."""
    result, _ = quad(f, a, b)
    return result

def compute_errors(a, b, n_values, exact_value):
    """Вычисляет погрешности для различных методов интегрирования."""
    errors = {
        'Left Rectangular': [],
        'Middle Rectangular': [],
        'Trapezoid': [],
        'Simpson': [],
        'Newton-Cotes': [],
        'Gauss': [],  # Добавляем метод Гаусса
    }

    for n in n_values:
        errors['Left Rectangular'].append(abs(left_rectangular(a, b, n) - exact_value))
        errors['Middle Rectangular'].append(abs(middle_rectangular(a, b, n) - exact_value))
        errors['Trapezoid'].append(abs(trapezoid(a, b, n) - exact_value))
        errors['Simpson'].append(abs(simpson(a, b, n) - exact_value))
        errors['Newton-Cotes'].append(abs(res_newton_cotes(a, b, n) - exact_value))

        # Вычисляем ошибки для метода Гаусса
        #nodes, weights = calculate_gauss_nodes_weights(n)  # Предположим, что у вас есть эта функция
        #errors['Gauss'].append(abs(gauss_quadrature(nodes, weights) - exact_value))

    return errors

def plot_errors(errors, n_values):
    """Строит график погрешностей для всех методов."""
    plt.figure(figsize=(10, 6))

    for method in ['Left Rectangular', 'Middle Rectangular', 'Trapezoid', 'Simpson', 'Gauss']:
        plt.plot(n_values, errors[method], marker='o', label=method)

    plt.yscale("log")
    plt.xlabel("Number of Subdivisions (n)")
    plt.ylabel("Absolute Error")
    plt.title("Comparison of Errors for Numerical Integration Methods")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_newton_cotes_error(errors, n_values):
    """Строит график погрешностей для метода Ньютона-Котеса."""
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, errors['Newton-Cotes'], marker='o', label="Newton-Cotes")
    plt.yscale("log")
    plt.xlabel("Number of Subdivisions (n)")
    plt.ylabel("Absolute Error")
    plt.title("Error of Newton-Cotes Method")
    plt.grid(True)
    plt.legend()
    plt.show()

def __main__():
    a = 0
    b = 2
    n_values = np.arange(10, 500, 5)

    exact_integral_value = calculate_exact_integral(a, b)
    errors = compute_errors(a, b, n_values, exact_integral_value)
    plot_errors(errors, n_values)
    plot_newton_cotes_error(errors, n_values)

if __name__ == "__main__":
    __main__()
