import numpy as np

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
    coeff = max(alpha, beta)
    moment_val = lambda x: x ** (degree - coeff + 1) / (degree - coeff + 1)

    return moment_val(b) - moment_val(a)

def cardano_formula(poly: np.ndarray) -> np.ndarray:
    """Формула Кардано для нахождения корней кубического уравнения"""
    q = (2 * poly[1] ** 3 / (54 * poly[0] ** 3)
         - poly[1] * poly[2] / (6 * poly[0] ** 2)
         + poly[3] / poly[0] / 2)

    p = (3 * poly[0] * poly[2] - poly[1] ** 2) / (9 * poly[0] ** 2)

    # Нас интересует только та часть формулы, где (q ** 2 + p ** 3) < 0. В методе Гаусса при разумном
    # числе разбиений, это всегда выполняется
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
    """Квадратичная формула Гаусса"""
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
        a_coeffs = np.linalg.solve(mu_matrix, -mu_vector).flatten()

        # Корни уравнения являются узлами
        poly_coeffs = np.append([1], a_coeffs[::-1])
        nodes = cardano_formula(poly_coeffs)

        # Решаем последнюю СЛАУ и получаем квадратурные коэффициенты
        nodes_matrix = np.array([nodes ** s for s in range(num_nodes)])

        mu_vector = moments[:num_nodes]
        A_coeffs = solve(nodes_matrix, mu_vector)

        result += A_coeffs @ func_(nodes)

    return result






# def compute_moments_gauss(a, z_i, z_i_1, alpha=0.6):
#     """Вычисляет моменты для метода Гаусса."""
#     mu_i0 = ((z_i - a) ** (1 - alpha) - (z_i_1 - a) ** (1 - alpha)) / (1 - alpha)
#     mu_i1 = ((z_i - a) ** (2 - alpha) - (z_i_1 - a) ** (2 - alpha)) / (2 - alpha) + a * mu_i0
#     mu_i2 = ((z_i - a) ** (3 - alpha) - (z_i_1 - a) ** (3 - alpha)) / (3 - alpha) + 2 * a * mu_i1 - a**2 * mu_i0

#     return mu_i0, mu_i1, mu_i2

# def compute_a_gauss(a, mu, n):
#     """
#     Решает систему линейных уравнений:
#     ∑_(j=0)^(n-1) a_j μ_(j+s) = -μ_(n+s), s=0,...,n-1.

#     :param a: Вектор коэффициентов a_j.
#     :param mu: Вектор значений моментов μ.
#     :param n: Размерность системы.
#     :return: Вектор решений.
#     """
#     A = np.zeros((n, n))
#     b = np.zeros(n)

#     for s in range(n):
#         for j in range(n):
#             A[s, j] = a[j] * mu[j + s]
#         b[s] = -mu[n + s]

#     solutions = np.linalg.solve(A, b)
#     return solutions

# def find_nodes(coefficients):
#     """
#     Находит узлы многочлена w(x) = 0.

#     :param coefficients: Список коэффициентов многочлена, начиная с a_n (высший степень).
#     :return: Корни (узлы) многочлена.
#     """
#     return np.roots(coefficients)

# def solve_linear_system(A, mu):
#     """
#     Решает систему линейных уравнений A @ x = mu.

#     :param A: Матрица коэффициентов.
#     :param mu: Вектор правых частей.
#     :return: Решение (узлы x_j).
#     """
#     return np.linalg.solve(A, mu)

# def gauss_quadrature(nodes, weights):
#     """
#     Вычисляет значение интеграла с помощью квадратурной формулы Гаусса.

#     :param nodes: Узлы (корни многочлена).
#     :param weights: Веса, соответствующие узлам.
#     :return: Значение интеграла.
#     """
#     integral = sum(weights[j] * F(nodes[j]) for j in range(len(nodes)))
#     return integral
