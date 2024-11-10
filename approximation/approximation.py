import numpy as np
import random
import math
import matplotlib.pyplot as plt

def f(x):
    x = np.asarray(x)
    return x * np.sqrt(x + 2)

def partition(a, b, n, k=5):
    points = np.linspace(a, b, n)
    points = np.repeat(points, k)
    return points.tolist()

def values(arg):
    arg = np.asarray(arg)
    if np.isscalar(arg):
        arg = np.array([arg])
    vals = f(arg)
    with_fault = vals + np.random.uniform(0, 0.1, size=vals.shape)
    return with_fault.tolist()

def lost(values, calculated_values):
    loss = np.sum((np.array(values) - np.array(calculated_values)) ** 2)
    return loss

def normal_mnk(x_values, y_values, n : int):
    Vand = np.array([[x ** j for j in range(n + 1)] for x in x_values])
    f = np.transpose(y_values)
    A = np.matmul(np.transpose(Vand ), Vand)
    b = np.matmul(np.transpose(Vand), f)

    coeffs = np.linalg.solve(A, b)

    return coeffs


def norm_evaluate(coeffs, x):
    """Вычисляет значение полинома с коэффициентами coeffs в точке или точках x."""
    x = np.asarray(x)  # Преобразование x в массив NumPy, если это не массив
    powers_of_x = np.column_stack([x ** i for i in range(len(coeffs))])
    return np.dot(powers_of_x, coeffs)

def orthogonal_polynomials(x, n):
    m = len(x)
    q = np.zeros((n+1, m))

    # Шаг 1: Задать q_0 и q_1
    q[0, :] = 1
    q[1, :] = x - np.mean(x)

    # Шаг 2: Вычислить q_j для j = 1, ..., n-1
    for j in range(1, n):
        alpha = np.sum(x * q[j]**2) / np.sum(q[j]**2)
        beta = np.sum(x * q[j] * q[j-1]) / np.sum(q[j-1]**2)
        q[j+1, :] = x * q[j] - alpha * q[j] - beta * q[j-1]

    return q

def least_squares_coefficients(x, y, n):
    q = orthogonal_polynomials(x, n)
    m = len(x)

    # Шаг 3: Вычислить коэффициенты a_k
    a = np.zeros(n+1)
    for k in range(n+1):
        a[k] = np.sum(q[k] * y) / np.sum(q[k]**2)

    return a, q

def ortho_evaluate(x, a, q):
    y_approx = np.zeros_like(x)
    for k in range(len(a)):
        y_approx += a[k] * q[k]
    return y_approx

x = partition(-1, 1, 100)
y = values(x)
n = 3

degrees = [1, 2, 3, 4, 5]
results = []

plt.figure(figsize=(15, 10))

for i, n in enumerate(degrees):
    # Нормальные уравнения
    coeff_norm = normal_mnk(x, y, n)
    y_approx_norm = norm_evaluate(coeff_norm, x)
    loss_norm = lost(y, y_approx_norm)

    # Ортогональные полиномы
    a, q = least_squares_coefficients(x, y, n)
    y_approx_ortho = ortho_evaluate(x, a, q)
    loss_ortho = lost(y, y_approx_ortho)

    # Запись результатов в таблицу
    results.append([n, loss_norm, loss_ortho])

    # Построение графиков для нормальных уравнений
    plt.subplot(5, 2, 2*i+1)
    plt.scatter(x, y, color='red', label='Экспериментальные точки')
    plt.plot(x, y_approx_norm, color='blue', label=f'Аппроксимация полиномом степени {n} (норм. уравнения)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Аппроксимация полиномом степени {n} (норм. уравнения)')
    plt.legend()

    # Построение графиков для ортогональных полиномов
    plt.subplot(5, 2, 2*i+2)
    plt.scatter(x, y, color='red', label='Экспериментальные точки')
    plt.plot(x, y_approx_ortho, color='green', label=f'Аппроксимация полиномом степени {n} (ортогон. полиномы)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Аппроксимация полиномом степени {n} (ортогон. полиномы)')
    plt.legend()

plt.tight_layout()
plt.show()

# Печать таблицы с результатами
print("Степень полинома | Сумма квадратов ошибок (МНК) | Сумма квадратов ошибок (ортогональные полиномы)")
for row in results:
    print(f"{row[0]} | {row[1]:.4f} | {row[2]:.4f}")
