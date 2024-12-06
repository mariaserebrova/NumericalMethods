import numpy as np
from numpy.linalg import solve

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

def moment(a: float, b: float, alpha: float, beta: float, degree: int) -> float:
    """Функция для вычисления моментов."""
    coeff = max(alpha, beta)
    moment_val = lambda x: x ** (degree - coeff + 1) / (degree - coeff + 1)
    return moment_val(b) - moment_val(a)

def cardano_formula(poly: np.ndarray) -> np.ndarray:
    """Формула Кардано для нахождения корней кубического уравнения."""
    q = (2 * poly[1] ** 3 / (54 * poly[0] ** 3)
         - poly[1] * poly[2] / (6 * poly[0] ** 2)
         + poly[3] / poly[0] / 2)

    p = (3 * poly[0] * poly[2] - poly[1] ** 2) / (9 * poly[0] ** 2)

    r = np.sign(q) * np.sqrt(np.abs(p))
    phi = np.arccos(q / r ** 3)

    y_roots = np.array([
        -2 * r * np.cos(phi / 3),
        2 * r * np.cos(np.pi / 3 - phi / 3),
        2 * r * np.cos(np.pi / 3 + phi / 3)
    ])

    roots = y_roots - poly[1] / poly[0] / 3
    return roots

def quad_gauss(func, a: float, b: float, alpha: float = 0, beta: float = 0, num_partitions: int = 100) -> float:
    """Квадратичная формула Гаусса."""
    if alpha == 0:  # Для удобства подсчета моментов, заменим переменную интегрирования
        bias, func_ = b, lambda x: func(bias - x)
    else:
        bias, func_ = a, lambda x: func(x + bias)

    a, b = 0, b - a
    result = 0
    num_nodes = 3

    for i in range(num_partitions):
        l_border = a + (b - a) * i / num_partitions
        r_border = a + (b - a) * (i + 1) / num_partitions

        # Вычисляем моменты 0,...,2n-1
        moments = np.array([moment(l_border, r_border, alpha, beta, i) for i in range(2 * num_nodes)])

        # Решим СЛАУ для нахождения коэффициентов уравнения
        n_range = np.arange(num_nodes)
        mu_matrix = moments[n_range.reshape(-1, 1) + n_range]
        mu_vector = moments[num_nodes:]
        a_coeffs = solve(mu_matrix, -mu_vector).flatten()

        # Корни уравнения являются узлами
        poly_coeffs = np.append([1], a_coeffs[::-1])
        nodes = cardano_formula(poly_coeffs)

        # Решаем последнюю СЛАУ и получаем квадратурные коэффициенты
        nodes_matrix = np.array([nodes ** s for s in range(num_nodes)])
        mu_vector = moments[:num_nodes]
        A_coeffs = solve(nodes_matrix, mu_vector)

        # Добавляем вклад текущего интервала в результат
        result += A_coeffs @ func_(nodes)

    return result
