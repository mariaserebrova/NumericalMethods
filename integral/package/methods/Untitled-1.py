def ranks(values, n=3):
    # Сортируем значения по убыванию, сохраняя исходные индексы
    sorted_indices = sorted(range(len(values)), key=lambda i: -values[i])

    # Инициализируем массив для рангов
    ranks = [0] * len(values)

    current_rank = 1
    i = 0
    while i < len(sorted_indices):
        # Находим все элементы с одинаковым значением начиная с текущего
        same_value_indices = [i]
        while i + 1 < len(sorted_indices) and values[sorted_indices[i]] == values[sorted_indices[i + 1]]:
            i += 1
            same_value_indices.append(i)

        # Средний ранг для группы одинаковых значений
        average_rank = sum(current_rank + j for j in range(len(same_value_indices))) / len(same_value_indices)

        # Присваиваем средний ранг всем элементам с одинаковым значением
        for idx in same_value_indices:
            ranks[sorted_indices[idx]] = average_rank

        # Обновляем текущий ранг для следующего уникального значения
        current_rank += len(same_value_indices)
        i += 1

    # Возвращаем первые n рангов в исходном порядке
    return ranks[:n]

# Пример использования функции
values = [6, 4, 7, 4.9, 7.3, 9.5, 8.3, 15, 9, 4, 8.4, 2, 5.1, 9, 4.3]
n = 7
print(ranks(values, n))  # Ожидаемый результат: [8.0, 3.5, 9.0, 10.0, 15.0, 2.0]
