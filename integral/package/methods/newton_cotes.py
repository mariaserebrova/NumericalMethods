import numpy as np
from numpy.linalg import solve
# from ..helpers import moment

def moment(a: float, b: float, alpha: float, beta: float, degree: int) -> float:
    coeff = max(alpha, beta)
    moment_val = lambda x: x ** (degree - coeff + 1) / (degree - coeff + 1)

    return moment_val(b) - moment_val(a)

def quad_newton_cotes(func, a: float, b: float, alpha: float = 0, beta: float = 0, num_partitions: int = 100):
    """Квадратичная формула Ньютона-Котса"""
    # Замена переменной интегрирования для удобства подсчета моментов
    if alpha == 0:
        bias, func_ = b, lambda x: func(bias - x)
    else:
        bias, func_ = a, lambda x: func(x + bias)

    a, b = 0, b - a  # Сдвиг границ для новой переменной

    result = 0
    num_nodes = 3  # Количество узлов

    for i in range(num_partitions):
        # Разбиение отрезка интегрирования
        nodes = [
            a + (b - a) * i / num_partitions,
            a + (b - a) * (i + 0.5) / num_partitions,
            a + (b - a) * (i + 1) / num_partitions
        ]

        # Вычисление моментов для текущего отрезка
        mu_vector = np.array([moment(nodes[0], nodes[2], alpha, beta, degree) for degree in range(num_nodes)])

        # Построение матрицы степеней узлов
        nodes = np.array(nodes)
        nodes_matrix = np.array([nodes ** s for s in range(num_nodes)])

        # Решение СЛАУ для нахождения весов квадратуры
        A_coeffs = solve(nodes_matrix, mu_vector)

        # Вычисление интегральной суммы
        result += A_coeffs @ func_(nodes)

    return result
