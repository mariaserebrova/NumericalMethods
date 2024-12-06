import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad as exact_quad

from package.methods.compound_quadrature_formulas import *
from package.methods.newton_cotes import *
from package.methods.gauss import *
from package.methods.richardson import *
from package.methods.compound_quadrature_formulas import quad


from tqdm import trange


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
        #errors['Newton-Cotes'].append(abs(res_newton_cotes(a, b, n) - exact_value))

        # Вычисляем ошибки для метода Гаусса
        #nodes, weights = calculate_gauss_nodes_weights(n)  # Предположим, что у вас есть эта функция
        #errors['Gauss'].append(abs(gauss_quadrature(nodes, weights) - exact_value))

    return errors


from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from colorama import Fore, Style


def red_string(var):
    """Возвращает текст красного цвета."""
    return Fore.RED + str(var) + Style.RESET_ALL


def green_string(var):
    """Возвращает текст зеленого цвета."""
    return Fore.GREEN + str(var) + Style.RESET_ALL


class Table:
    """Класс для работы с таблицами, удобного вывода и обновления данных."""
    def __init__(self, columns):
        self.__column_names = columns
        self.__rows = [[np.inf for _ in range(len(columns))] for _ in range(3)]

    def update_column(self, column: str or int, values: tuple[float, float, float]):
        """Обновление данных в конкретной колонке."""
        idx = self.__column_names.index(column)

        if values[2] < self.__rows[2][idx]:
            self.__rows[0][idx] = values[0]
            self.__rows[1][idx] = values[1]
            self.__rows[2][idx] = values[2]

    def show(self, highlight_best: bool = False):
        """Отображение таблицы с подсветкой наилучшего результата."""
        column_names = deepcopy(self.__column_names)
        rows = deepcopy(self.__rows)

        if highlight_best:
            best_column_idx = rows[2].index(min(rows[2]))

            column_names[best_column_idx] = green_string(column_names[best_column_idx])
            rows[0][best_column_idx] = green_string(rows[0][best_column_idx])
            rows[1][best_column_idx] = green_string(rows[1][best_column_idx])
            rows[2][best_column_idx] = green_string(rows[2][best_column_idx])

        quadratic_info_table = PrettyTable([''] + column_names)
        quadratic_info_table.add_row(['Partitions'] + rows[0])
        quadratic_info_table.add_row(['Value'] + rows[1])
        quadratic_info_table.add_row(['Residue'] + rows[2])

        print(quadratic_info_table)


def func(x: float or np.ndarray) -> float or np.ndarray:
    """Определение целевой функции."""
    return 3 * np.cos(3.5 * x) * np.exp(4 * x / 3) + 2 * np.sin(3.5 * x) * np.exp(-2 * x / 3) + 4 * x


def moment(a: float, b: float, alpha: float, beta: float, degree: int) -> float:
    """Вычисление момента распределения."""
    coeff = max(alpha, beta)
    moment_val = lambda x: x ** (degree - coeff + 1) / (degree - coeff + 1)

    return moment_val(b) - moment_val(a)


def cardano_formula(poly: np.ndarray) -> np.ndarray:
    """Реализация формулы Кардано для решения кубических уравнений."""
    q = (2 * poly[1] ** 3 / (54 * poly[0] ** 3)
         - poly[1] * poly[2] / (6 * poly[0] ** 2)
         + poly[3] / poly[0] / 2)

    p = (3 * poly[0] * poly[2] - poly[1] ** 2) / (9 * poly[0] ** 2)

    # Рассматривается случай, когда (q ** 2 + p ** 3) < 0
    r = np.sign(q) * np.sqrt(np.abs(p))
    phi = np.arccos(q / r ** 3)

    y_roots = np.array([
        -2 * r * np.cos(phi / 3),
        2 * r * np.cos(np.pi / 3 - phi / 3),
        2 * r * np.cos(np.pi / 3 + phi / 3)
    ])

    roots = y_roots - poly[1] / poly[0] / 3
    return roots


def plot_residues_graphics(methods_residues: dict[str, list[float]]):
    """Визуализация графиков погрешностей."""
    plt.figure(figsize=(15, 6))

    # Обычная шкала
    plt.subplot(1, 2, 1)
    plt.title('График погрешности')
    for method, residues in methods_residues.items():
        plt.plot(range(len(residues)), residues, label=f'{method} Residues')

    plt.ylabel('Погрешность')
    plt.xlabel('Разбиения')
    plt.grid()
    plt.legend()

    # Логарифмическая шкала
    plt.subplot(1, 2, 2)
    plt.title('График погрешности (в логарифмической шкале)')
    for method, residues in methods_residues.items():
        plt.plot(range(len(residues)), np.log(np.array(residues)), label=f'{method} Residues')

    plt.ylabel('Логарифм погрешности')
    plt.xlabel('Разбиения')
    plt.grid()
    plt.legend()

    plt.show()

# def plot_residues_graphics(methods_residues: dict[str, list[float]], y_scale_limit: float = 1e-3):
#     """Визуализация графиков погрешностей с меньшим масштабом оси Y."""
#     plt.figure(figsize=(15, 6))

#     # Обычная шкала
#     plt.subplot(1, 2, 1)
#     plt.title('График погрешности')
#     for method, residues in methods_residues.items():
#         plt.plot(range(len(residues)), residues, label=f'{method} Residues')

#     plt.ylabel('Погрешность')
#     plt.xlabel('Разбиения')
#     plt.grid()
#     plt.legend()
#     plt.ylim(0, y_scale_limit)  # Устанавливаем ограничение по оси Y

#     # Логарифмическая шкала
#     plt.subplot(1, 2, 2)
#     plt.title('График погрешности (в логарифмической шкале)')
#     for method, residues in methods_residues.items():
#         plt.plot(range(len(residues)), np.log(np.clip(residues, 1e-12, None)), label=f'{method} Residues')

#     plt.ylabel('Логарифм погрешности')
#     plt.xlabel('Разбиения')
#     plt.grid()
#     plt.legend()

#     plt.show()

#------------------------------

# def plot_errors(errors, n_values):
#     """Строит график погрешностей для всех методов."""
#     plt.figure(figsize=(10, 6))

#     for method in ['Left Rectangular', 'Middle Rectangular', 'Trapezoid', 'Simpson', 'Gauss']:
#         plt.plot(n_values, errors[method], marker='o', label=method)

#     plt.yscale("log")
#     plt.xlabel("Number of Subdivisions (n)")
#     plt.ylabel("Absolute Error")
#     plt.title("Comparison of Errors for Numerical Integration Methods")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# def plot_newton_cotes_error(errors, n_values):
#     """Строит график погрешностей для метода Ньютона-Котеса."""
#     plt.figure(figsize=(8, 5))
#     plt.plot(n_values, errors['Newton-Cotes'], marker='o', label="Newton-Cotes")
#     plt.yscale("log")
#     plt.xlabel("Number of Subdivisions (n)")
#     plt.ylabel("Absolute Error")
#     plt.title("Error of Newton-Cotes Method")
#     plt.grid(True)
#     plt.legend()
#     plt.show()
def main():
    a = 0
    b = 2
    alpha = 0
    beta = 0.6
    methods = ['Left Rectangle', 'Right Rectangle', 'Middle Rectangle', 'Trapezia', 'Simpson', 'Gauss', 'Newton Cotes']

    print('∘ЗАДАНИЕ 1: Вычисление интегралов при помощи составных квадратурных формул.')
    real_value, *_ = exact_quad(func, a, b)
    real_value_weight, *_ = exact_quad(lambda x: func(x) / (x - a) ** alpha / (b - x) ** beta, a, b)

    methods_residues = {mtd: [] for mtd in methods}  # Список погрешностей методов
    table = Table(methods)  # Таблица лучших значений методов

    for n_part in trange(1, 101, desc='Вычисление'):  # Будем увеличивать разбиение и смотреть на поведение погрешности
        for method in methods:
            calc_value = quad(func, a, b, method=method, alpha=alpha, beta=beta, num_partitions=n_part)
            residue = abs(real_value_weight - calc_value) if method in ['Gauss', 'Newton Cotes'] \
                else abs(real_value - calc_value)

            table.update_column(column=method, values=(n_part, calc_value, residue))
            methods_residues[method].append(residue)

    # Таблица погрешностей методов
    table.show(highlight_best=True)

    # Строим графики погрешностей
    plot_residues_graphics(methods_residues)

    print('\n∘ЗАДАНИЕ 2: Методы оценки составных квадратурных формул.')
    # Погрешность для метода Ньютона-Котса
    nc_quad = lambda n: quad(func, a, b, alpha=alpha, beta=beta, method='Newton Cotes', num_partitions=n)
    nc_residue, nc_step, nc_partition = richardson(nc_quad, gap_len=b - a, min_part=3, eps=10 ** -6)

    print(red_string('Метод Ньютона-Котса'))
    print(f'Длина шага разбиения: {green_string(nc_step)}')
    print(f'Разбиение: {green_string(nc_partition)} точек')
    print(f'Погрешность: {green_string(nc_residue)}\n')

    # Погрешность для метода Гаусса
    gs_quad = lambda n: quad(func, a, b, alpha=alpha, beta=beta, method='Gauss', num_partitions=n)
    gs_residue, gs_step, gs_partition = richardson(gs_quad, gap_len=b - a, min_part=3, eps=10 ** -6)

    print(red_string('Метод Гаусса'))
    print(f'Длина шага разбиения: {green_string(gs_step)}')
    print(f'Разбиение: {green_string(gs_partition)} точек')
    print(f'Погрешность: {green_string(gs_residue)}\n')


if __name__ == '__main__':
    main()
