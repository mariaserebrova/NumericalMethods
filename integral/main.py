import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return(3.5 * np.cos(1.5 * x) * np.exp(x / 4) + 4 * np.sin(3.5 * x) * np.exp(-1 * 3 * x) + 4 * x)

def p(x, a=2.5, b=3.3, alpha=2/3, beta=0):
    if x <= a or x >= b:
        return 0  # возвращаем 0, если x выходит за пределы
    return (x - a) ** (-1 * alpha) * (b - x) ** (-1 * beta)

def F(x):
    return f(x) * p(x)

def calculate_exact_integral(a, b):
    result, _ = quad(f, a, b)  # Используем quad из scipy для вычисления интеграла
    return result

def compute_errors(a, b, n_values, exact_value):
    errors = {
        'Left Rectangular': [],
        'Middle Rectangular': [],
        'Trapezoid': [],
        'Simpson': [],
        'Newton-Cotes': []
    }

    for n in n_values:
        errors['Left Rectangular'].append(abs(left_rectangular(a, b, n) - exact_value))
        errors['Middle Rectangular'].append(abs(middle_rectangular(a, b, n) - exact_value))
        errors['Trapezoid'].append(abs(trapezoid(a, b, n) - exact_value))
        errors['Simpson'].append(abs(simpson(a, b, n) - exact_value))
        errors['Newton-Cotes'].append(abs(res_newton_cotes(a, b, n) - exact_value))

    return errors

def plot_errors(errors, n_values):
    plt.figure(figsize=(10, 6))

    for method in ['Left Rectangular', 'Middle Rectangular', 'Trapezoid', 'Simpson']:
        plt.plot(n_values, errors[method], marker='o', label=method)

    plt.yscale("log")
    plt.xlabel("Number of Subdivisions (n)")
    plt.ylabel("Absolute Error")
    plt.title("Comparison of Errors for Numerical Integration Methods")
    plt.grid(True)
    plt.legend()

    plt.show()

def plot_newton_cotes_error(errors, n_values):
    plt.figure(figsize=(8, 5))

    plt.plot(n_values, errors['Newton-Cotes'], marker='o', label="Newton-Cotes")

    plt.yscale("log")
    plt.xlabel("Number of Subdivisions (n)")
    plt.ylabel("Absolute Error")
    plt.title("Error of Newton-Cotes Method")
    plt.grid(True)
    plt.legend()

    plt.show()


def compute_moments_newton_cotes(a, z_i, z_i_1, alpha = 0.6):
    mu_i0 = ((z_i - a) ** (1 - alpha) - (z_i_1 - a) ** (1 - alpha)) / (1 - alpha)
    mu_i1 = ((z_i - a) ** (2 - alpha) - (z_i_1 - a) ** (2 - alpha)) / (2 - alpha) + a * mu_i0
    mu_i2 = ((z_i - a) ** (3 - alpha) - (z_i_1 - a) ** (3 - alpha)) / (3 - alpha) + 2 * a * mu_i1 - a**2 * mu_i0

    return mu_i0, mu_i1, mu_i2

def compute_weights_newton_cotes(a, b, z_i, z_i_1, alpha = 0, beta = 0.6):
    z_i_half = (z_i_1 + z_i) / 2

    mu_i0, mu_i1, mu_i2 = compute_moments_newton_cotes(a, z_i, z_i_1, alpha)

    A_i1 = (mu_i2 - mu_i1 * (z_i_half + z_i) + mu_i0 * z_i_half * z_i) / ((z_i_half - z_i_1) * (z_i - z_i_half))
    A_i2 =  -1 * ((mu_i2 - mu_i1 * (z_i_1 + z_i) + mu_i0 * z_i_1 * z_i) / (z_i_half - z_i_1) * (z_i - z_i_half))
    A_i3 = (mu_i2 - mu_i1 * (z_i_half + z_i_1) + mu_i0 * z_i_half * z_i_1) / (z_i - z_i_half) * (z_i - z_i_1)

    return A_i1, A_i2, A_i3

import numpy as np

# решение системы линейных уравнений для нахождения коэффициентов A_j
def solve_linear_system(x_values, mu_values):
    n = len(x_values)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i, j] = x_values[j] ** i  # степени x_j

    A_values = np.linalg.solve(matrix, mu_values)
    return A_values

def res_newton_cotes(a, b, n, alpha=0, beta=0.6):
    z = np.linspace(a, b, n)

    mu_values = [compute_moments_newton_cotes(a, z[i], z[i-1], alpha)[0] for i in range(1, n)]

    A_values = solve_linear_system(z[:-1], mu_values)

    res = 0
    for i in range(1, n):
        z_i = z[i]
        z_i_1 = z[i - 1]

        f_z_i_1 = F(z_i_1)
        f_z_i = F(z_i)
        f_z_i_half = F((z_i_1 + z_i) / 2)

        res += A_values[i-1] * (f_z_i_1 + f_z_i + f_z_i_half)

    return res

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
