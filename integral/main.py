import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad as exact_quad

from package.methods.compound_quadrature_formulas import *
from package.methods.newton_cotes import *
from package.methods.gauss import *
from package.methods.richardson import *
from package.methods.compound_quadrature_formulas import quad

from tqdm import trange
from copy import deepcopy
from prettytable import PrettyTable


def f(x):
    """Функция для интегрирования."""
    return (3 * np.cos(1.5 * x) * np.exp(x / 4) +
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

            column_names[best_column_idx] = f"*{column_names[best_column_idx]}*"
            rows[0][best_column_idx] = f"*{rows[0][best_column_idx]}*"
            rows[1][best_column_idx] = f"*{rows[1][best_column_idx]}*"
            rows[2][best_column_idx] = f"*{rows[2][best_column_idx]}*"

        print("Table Overview:\n")
        for idx, col_name in enumerate(column_names):
            print(f"{col_name}:")
            print(f"  Partitions: {rows[0][idx]}")
            print(f"  Value:      {rows[1][idx]}")
            print(f"  Residue:    {rows[2][idx]}")
            print("-" * 30)


def func(x: float or np.ndarray) -> float or np.ndarray:
    """Определение целевой функции."""
    return 3 * np.cos(1.5 * x) * np.exp(x / 4) + 4 * np.sin(3.5 * x) * np.exp(-3 * x) + 4 * x


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

def main():
    a = 2.5
    b = 3.3
    alpha = 2 / 3
    beta = 0
    methods = ['Left Rectangle', 'Right Rectangle', 'Middle Rectangle', 'Trapezia', 'Simpson', 'Gauss', 'Newton Cotes']

    print('∘ЗАДАНИЕ 1: Вычисление интегралов при помощи составных квадратурных формул.')
    real_value, *_ = exact_quad(func, a, b)
    real_value_weight, *_ = exact_quad(F, a, b)

    methods_residues = {mtd: [] for mtd in methods}
    table = Table(methods)

    for n_part in trange(1, 101, desc='Вычисление'):
        for method in methods:
            calc_value = quad(func, a, b, method=method, alpha=alpha, beta=beta, num_partitions=n_part)
            residue = abs(real_value_weight - calc_value) if method in ['Gauss', 'Newton Cotes'] \
                else abs(real_value - calc_value)

            table.update_column(column=method, values=(n_part, calc_value, residue))
            methods_residues[method].append(residue)

    table.show(highlight_best=True)
    plot_residues_graphics(methods_residues)

    print('\n∘ЗАДАНИЕ 2: Методы оценки составных квадратурных формул.')
    nc_quad = lambda n: quad(func, a, b, alpha=alpha, beta=beta, method='Newton Cotes', num_partitions=n)
    nc_residue, nc_step, nc_partition = richardson(nc_quad, gap_len=b - a, min_part=3, eps=10 ** -6)

    print('Метод Ньютона-Котса')
    print(f'Длина шага разбиения: {nc_step}')
    print(f'Разбиение: {nc_partition} точек')
    print(f'Погрешность: {nc_residue}\n')

    gs_quad = lambda n: quad(func, a, b, alpha=alpha, beta=beta, method='Gauss', num_partitions=n)
    gs_residue, gs_step, gs_partition = richardson(gs_quad, gap_len=b - a, min_part=3, eps=10 ** -6)

    print('Метод Гаусса')
    print(f'Длина шага разбиения: {gs_step}')
    print(f'Разбиение: {gs_partition} точек')
    print(f'Погрешность: {gs_residue}\n')


if __name__ == '__main__':
    main()
