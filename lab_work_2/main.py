import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
import math
import statistics

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор случайных чисел")
        self.root.geometry("500x600")
        
        self.data = []
        
        self.create_widgets()
        
    def create_widgets(self):
        # Раздел для генерации данных
        generate_frame = ttk.LabelFrame(self.root, text="Генерация данных", padding=10)
        generate_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Параметры генерации
        ttk.Label(generate_frame, text="Количество чисел (N):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.n_entry = ttk.Entry(generate_frame, width=15)
        self.n_entry.grid(row=0, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        self.n_entry.insert(0, "100")
        
        ttk.Label(generate_frame, text="Начало интервала (a):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.a_entry = ttk.Entry(generate_frame, width=15)
        self.a_entry.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        self.a_entry.insert(0, "0")
        
        ttk.Label(generate_frame, text="Конец интервала (b):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.b_entry = ttk.Entry(generate_frame, width=15)
        self.b_entry.grid(row=2, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        self.b_entry.insert(0, "10")
        
        # Кнопки генерации и загрузки
        button_frame = ttk.Frame(generate_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Сгенерировать данные", 
                  command=self.generate_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Сохранить в файл", 
                  command=self.save_to_file).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(button_frame, text="Загрузить из файла", 
                  command=self.load_from_file).pack(side=tk.LEFT)
        
        # Фрейм для результатов
        results_frame = ttk.LabelFrame(self.root, text="Статистические параметры", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создание меток для результатов
        self.result_labels = {}
        
        parameters = [
            "Количество чисел:",
            "Максимальное значение:",
            "Минимальное значение:", 
            "Среднее значение:",
            "Медиана:",
            "Мода(ы):",
            "Вероятность случайного выбора:",
            "Стандартное отклонение:",
            "Промах (норм. распределение):",
            "Промах (метод 3-сигма):"
        ]
        
        for i, param in enumerate(parameters):
            ttk.Label(results_frame, text=param, font=('Arial', 9, 'bold')).grid(
                row=i, column=0, sticky=tk.W, pady=2, padx=(0, 10))
            
            value_label = ttk.Label(results_frame, text="-", foreground="black")
            value_label.grid(row=i, column=1, sticky=tk.W, pady=2)
            self.result_labels[param] = value_label
        
        # Кнопка анализа
        ttk.Button(results_frame, text="Выполнить анализ", 
                  command=self.analyze_data).grid(row=len(parameters), column=0, columnspan=2, pady=10)
        
    def generate_data(self):
        try:
            N = int(self.n_entry.get())
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            
            if b <= a:
                messagebox.showerror("Ошибка", "Конец интервала должен быть больше начала!")
                return
            
            if N <= 0:
                messagebox.showerror("Ошибка", "Количество чисел должно быть положительным!")
                return
                
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения!")
            return
        
        # Генерация случайных чисел
        self.data = [random.uniform(a, b) for _ in range(N)]
        messagebox.showinfo("Успех", f"Сгенерировано {N} чисел в интервале [{a}, {b}]")
    
    def save_to_file(self):
        if not self.data:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения!")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Сохранить данные в файл"
        )
    
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as file:
                    for number in self.data:
                        file.write(f"{number:.6f}\n")
                messagebox.showinfo("Успех", f"Данные сохранены в файл:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")

    def load_from_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as file:
                    self.data = [float(line.strip()) for line in file if line.strip()]
                
                messagebox.showinfo("Успех", f"Загружено {len(self.data)} чисел из файла")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
    
    def calculate_confidence_interval(self, mean_value, std_dev, confidence_level = 0.95):
        if not self.data or len(self.data) < 2:
            return None, None, 0
        
        n = len(self.data)
        
        # Приближенные значения t-распределения для разных уровней доверия
        t_values = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576
        }
        
        # Используем z-value для больших выборок (n > 30) как приближение
        if n > 30:
            t_value = t_values.get(confidence_level, 1.960)
        else:
            # Для малых выборок используем более консервативные значения
            small_sample_t = {
                0.90: 2.920,  # для n=2
                0.95: 4.303,
                0.99: 9.925
            }
            t_value = small_sample_t.get(confidence_level, 4.303)
        
        # Стандартная ошибка среднего
        standard_error = std_dev / (n ** 0.5)
        
        # Допустимая погрешность
        margin_of_error = t_value * standard_error
           
        return margin_of_error 


    def find_outliers_normal(self, mean, std_dev):
        if std_dev == 0:
            return []
        
        # Для нормального распределения считаем промахами значения за пределами 2.5 сигма
        margin_of_error = self.calculate_confidence_interval(mean, std_dev)
        for x in self.data:
            if x < mean - margin_of_error or x > mean + margin_of_error:
                return x

        return None
    
    def find_outliers_3sigma(self, mean, std_dev):
        if std_dev == 0:
            return []
        
        # По методу 3-сигма считаем промахами значения за пределами 3 сигма
        threshold = 3 * std_dev

        for x in self.data:
            if x < mean - threshold or x > mean + threshold:
                return outlier

        return None
        
    def analyze_data(self):
        if not self.data:
            messagebox.showwarning("Предупреждение", "Нет данных для анализа!")
            return
        
        # Вычисление статистики
        
        max_value = max(self.data)
        min_value = min(self.data)
        
        # Среднее значение

        mean_value = statistics.mean(self.data)

        # Медиана

        median_value = statistics.median(self.data)
        
        # Мода

        mode_values = statistics.multimode(self.data)

        if len(mode_values) == len(self.data):
            mode_result = "Все значения уникальны (моды нет)"
        else:
            mode_result = ", ".join([f"{x:.6f}" for x in mode_values])
        
        # Вероятность (для равномерного распределения)
        probability = 1 / len([i for i in self.data if i % 2 != 0])
        
        # Стандартное отклонение
        std_dev = statistics.stdev(self.data)
        
        # Поиск промахов
        outliers_normal = self.find_outliers_normal(mean_value, std_dev)
        outliers_3sigma = self.find_outliers_3sigma(mean_value, std_dev)
        
        # Форматирование вывода промахов
        if outliers_normal:
            normal_outliers_str = f"{outliers_normal}"
        else:
            normal_outliers_str = "Нет промахов"
            
        if outliers_3sigma:
            sigma_outliers_str = f"{outliers_3sigma}"
        else:
            sigma_outliers_str = "Нет промахов"
        
        # Обновление меток с результатами
        self.result_labels["Количество чисел:"].config(text=f"{len(self.data)}")
        self.result_labels["Максимальное значение:"].config(text=f"{max_value:.6f}")
        self.result_labels["Минимальное значение:"].config(text=f"{min_value:.6f}")
        self.result_labels["Среднее значение:"].config(text=f"{mean_value:.6f}")
        self.result_labels["Медиана:"].config(text=f"{median_value:.6f}")
        self.result_labels["Мода(ы):"].config(text=mode_result)
        self.result_labels["Вероятность случайного выбора:"].config(text=f"{probability:.6f}")
        self.result_labels["Стандартное отклонение:"].config(text=f"{std_dev:.6f}")
        self.result_labels["Промах (норм. распределение):"].config(text=normal_outliers_str)
        self.result_labels["Промах (метод 3-сигма):"].config(text=sigma_outliers_str)

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
