import random
import statistics
from collections import Counter

def main():

    # Ввод и проверка на корректность ввода значении с клавиатуры

    try:
        N = int(input("Введите количество измерений (N): "))
        a = float(input("Введите начало интервала (a): "))
        b = float(input("Введите конец интервала (b): "))
        
        if b <= a:
            print("Ошибка: конец интервала должен быть больше начала!")
            return
        
        if N <= 0:
            print("Ошибка: количество измерений должно быть положительным!")
            return
            
    except ValueError:
        print("Ошибка: введите корректные числовые значения!")
        return

    # Генератор случайных чисел
   
    random_numbers = [random.uniform(a, b) for _ in range(N)]
    
    # Создание/открытие и запись случайных чисел в файл 

    filename = f"numbers.txt"
    with open(filename, 'w') as file:
        for number in random_numbers:
            file.write(f"{number:.6f}\n")
        
    # Определение минимального и максимального значения из выборки

    max_value = max(random_numbers)
    min_value = min(random_numbers)

    # Определение среднего значения

    mean_value = statistics.mean(random_numbers)

    # Определение медианы

    median_value = statistics.median(random_numbers)
    
    # Определение моды

    mode_values = statistics.multimode(random_numbers)
    if len(mode_values) == len(random_numbers):
        mode_result = "Все значения уникальны (моды нет)"
    else:
        mode_result = ", ".join([f"{x:.6f}" for x in mode_values])
   
    # Определение вероятности выбора случайного числа из выборки

    probability = 1 / (b - a)
    
    # Результаты

    print(f"Максимальное значение: {max_value:.6f}")
    print(f"Минимальное значение: {min_value:.6f}")
    print(f"Среднее значение: {mean_value:.6f}")
    print(f"Мода(ы): {mode_result}")
    print(f"Медиана: {median_value:.6f}")
    print(f"Вероятность выбора случайного числа: {probability:.6f}")

# Проверяет, является ли текущий скрипт основным 

if __name__ == "__main__":
    main()
