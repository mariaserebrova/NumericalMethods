import numpy as np

def richardson(quad, gap_len: float, min_part: int = 1, max_part: int = np.inf, eps: float = 1e-12):
    """Метод Ричардсона для оценки погрешностей интегралов"""
    r, R = 2, np.inf  # Инициализация количества шагов и погрешности
    best_step, best_part = 0, 0

    while R > eps and (2 ** r) * min_part <= max_part:
        # Вычисление значений интегралов при разном числе разбиений
        values = [quad(n=2 ** i * min_part) for i in range(r + 1)]

        # Проверка на то, что значения не совпадают, чтобы избежать деления на ноль
        if abs(values[-1] - values[-2]) < 1e-12 or abs(values[-2] - values[-3]) < 1e-12:
            break

        # Формула Эйткена для оценки скорости сходимости
        m = -np.log(abs((values[-1] - values[-2]) / (values[-2] - values[-3]))) / np.log(2)

        # Шаги сетки
        steps = [gap_len / (2 ** i * min_part) for i in range(r)]

        # Построение системы уравнений для уточнения значения
        steps_matrix = np.array([[step ** i for i in range(r)] for step in steps])
        values_vector = np.array(values[:r])

        # Решение СЛАУ для уточненного интеграла
        try:
            J, *_ = np.linalg.solve(steps_matrix, values_vector)
        except np.linalg.LinAlgError:
            print("Ошибка при решении СЛАУ. Матрица может быть вырождена.")
            break

        # Текущая погрешность
        cur_R = abs(J - values[-1])
        if cur_R < R:
            best_part = int(gap_len / steps[-1])
            best_step = steps[-1]
            R = cur_R

        r += 1

    return R, best_step, best_part
