import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

def generate_matrix_A(N, debug_mode=False, debug_type='sequential', filename=None):
    """Генерация матрицы A с различными вариантами отладочного заполнения"""
    if not debug_mode:
        A = np.random.randint(-10, 11, size=(N, N))
        print("Используется случайное заполнение")
        return A
    else: 
        if debug_type == 'sequential':
            A = np.arange(N*N, dtype=float).reshape(N, N) - 10
            print("Используется последовательное заполнение")
        elif debug_type == 'file_input':
            A = read_matrix_from_file(filename, N)
            print("Используется ввод из файла")
        return A

def read_matrix_from_file(filename, N):
    try:
        if filename and os.path.exists(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()
                matrix_data = []
                for line in lines:
                    row = [float(x) for x in line.strip().split()]
                    matrix_data.append(row)
                
                if len(matrix_data) >= N and len(matrix_data[0]) >= N:
                    A = np.array(matrix_data[:N])[:, :N]
                    print(f"Матрица успешно загружена из файла {filename}")
                    return A
                else:
                    print("Файл содержит матрицу недостаточного размера. Используется последовательное заполнение.")
                    return np.arange(N*N, dtype=float).reshape(N, N) - 10
        else:
            print("Файл не найден. Используется последовательное заполнение.")
            return np.arange(N*N, dtype=float).reshape(N, N) - 10
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}. Используется последовательное заполнение.")
        return np.arange(N*N, dtype=float).reshape(N, N) - 10

def create_zeros_test_matrix(N):
    A = np.ones((N, N)) * 2
    
    mid = N // 2
    for i in range(mid, N):
        for j in range(mid, N):
            if (j - mid) % 2 == 0:
                A[i, j] = 0
    
    for i in range(mid, N):
        if (i - mid) % 2 == 1:
            for j in range(mid, mid + 1):
                A[i, j] = -1
    
    return A

def create_negative_test_matrix(N):
    A = np.ones((N, N)) * 3 
    
    mid = N // 2
    for i in range(mid, N):
        if (i - mid) % 2 == 1:
            for j in range(mid, N):
                A[i, j] = -5
    
    for i in range(mid, mid + 1):
        for j in range(mid, mid + 2):
            if (j - mid) % 2 == 0:
                A[i, j] = 0
    
    return A

def create_determinant_test_matrix(N):
    A = np.eye(N) * 10

    for i in range(N):
        for j in range(N):
            if i != j:
                A[i, j] = 1
    return A

def create_symmetric_matrix(N):
    A = np.random.randint(-5, 6, size=(N, N))
    return (A + A.T) // 2

def extract_submatrices(A):
    N = A.shape[0]
    mid = N // 2
    
    B = A[:mid, :mid]
    C = A[:mid, mid:]
    D = A[mid:, :mid]
    E = A[mid:, mid:]
    
    return B, C, D, E

def check_condition(E):
    zero_in_odd_cols = np.sum(E[:, 1::2] == 0)
    
    neg_in_even_rows = np.sum(E[1::2, :] < 0)
    
    return zero_in_odd_cols > neg_in_even_rows

def form_matrix_F(A, condition):
    F = A.copy()
    N = A.shape[0]
    mid = N // 2
    
    B, C, D, E = extract_submatrices(A)
    
    if condition:
        F[:mid, :mid] = np.flip(C, axis=1)
        F[:mid, mid:] = np.flip(B, axis=1)
        print("Поменяли местами C и B симметрично")
    else:
        # Меняем B и E несимметрично
        F[:mid, :mid] = E  # B заменяем на E
        F[mid:, mid:] = B  # E заменяем на B
        print("Поменяли местами B и E несимметрично")
    
    return F

def compute_expression(A, F, K, condition):
    try:
        if condition:
            A_inv = np.linalg.inv(A)
            A_trans = A.T
            result = A_inv @ A_trans - K * F
            print("Вычислено: A^(-1) * A^T - K * F")
        else:
            A_trans = A.T
            G = np.tril(A)
            F_inv = np.linalg.inv(F)
            result = (A_trans + G - F_inv) * K
            print("Вычислено: (A^T + G - F^(-1)) * K")
        
        return result
    except np.linalg.LinAlgError:
        print("Ошибка: матрица вырожденная, невозможно вычислить обратную матрицу")
        return np.zeros_like(A)

def plot_matrices(A, F, result):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    im1 = axes[0, 0].imshow(A, cmap='coolwarm', interpolation='nearest')
    axes[0, 0].set_title('Матрица A - Тепловая карта')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0, 0])
    
    X, Y = np.meshgrid(range(F.shape[1]), range(F.shape[0]))
    axes[0, 1].remove()
    ax3d = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax3d.plot_surface(X, Y, F, cmap=cm.viridis, alpha=0.8)
    ax3d.set_title('Матрица F - 3D поверхность')
    plt.colorbar(surf, ax=ax3d, shrink=0.5)
    
    axes[1, 0].bar(range(result.size), result.flatten(), alpha=0.7)
    axes[1, 0].set_title('Результат вычислений - Столбчатая диаграмма')
    axes[1, 0].set_xlabel('Индекс элемента')
    axes[1, 0].set_ylabel('Значение')
    axes[1, 0].grid(True, alpha=0.3)
    
    diag_A = np.diag(A)
    diag_F = np.diag(F)
    x = range(len(diag_A))
    axes[1, 1].plot(x, diag_A, 'bo-', label='Диагональ A', markersize=4)
    axes[1, 1].plot(x, diag_F, 'ro-', label='Диагональ F', markersize=4)
    axes[1, 1].set_title('Сравнение диагоналей A и F')
    axes[1, 1].set_xlabel('Индекс')
    axes[1, 1].set_ylabel('Значение')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def select_debug_mode():
    print("\nРежимы:")
    print("1. Случайное заполнение")
    print("2. (Отладка) Последовательное заполнение")
    print("3. (Отладка) Ввод из файла")
    
    choice = input("Выберите режим (1-3, по умолчанию 1): ").strip()
    
    debug_mode = False
    debug_type = None
    filename = None
    
    match choice:
        case '2':
            debug_mode = True
            debug_type = "sequential"
        case '3':
            debug_mode = True
            debug_type = 'file_input'
            filename = input("Введите имя файла: ").strip()
            if not filename:
                filename = 'matrix.txt'
    
    return debug_mode, debug_type, filename

def create_example_file(N):
    filename = 'matrix_example.txt'
    with open(filename, 'w') as f:
        for i in range(N):
            row = [i*N + j for j in range(N)]
            f.write(' '.join(map(str, row)) + '\n')
    print(f"Создан пример файла {filename} с матрицей {N}x{N}")

def main():
    K = float(input("Введите число K: "))
    N = int(input("Введите размер матрицы N (четное число): "))
    
    if N % 2 != 0:
        print("N должно быть четным числом! Увеличиваю N на 1.")
        N += 1
    
    debug_mode, debug_type, filename = select_debug_mode()
    
    if debug_type == 'file_input' and not os.path.exists(filename):
        print(f"Файл {filename} не найден.")
        create = input("Создать пример файла? (y/n): ").lower()
        if create == 'y':
            create_example_file(N)
            filename = 'matrix_example.txt'
    
    A = generate_matrix_A(N, debug_mode, debug_type, filename)
    
    """
    print("\nМатрица A:")
    print(A)
    """
    
    B, C, D, E = extract_submatrices(A)
    
    """
    print("\nПодматрица B:")
    print(B)
    print("\nПодматрица C:")
    print(C)
    print("\nПодматрица D:")
    print(D)
    print("\nПодматрица E:")
    print(E)
    """

    condition_E = check_condition(E)
    zero_count = np.sum(E[:, 1::2] == 0)
    neg_count = np.sum(E[1::2, :] < 0)

    """
    print(f"\nВ подматрице E:")
    print(f"Количество нулей в нечетных столбцах: {zero_count}")
    print(f"Количество отрицательных в четных строках: {neg_count}")
    print(f"Условие (нули > отрицательные): {condition_E}")
    """

    F = form_matrix_F(A, condition_E)

    """
    print("\nМатрица F:")
    print(F)
    """

    det_A = np.linalg.det(A)
    sum_diag_F = np.trace(F)
    condition_det = det_A > sum_diag_F
    
    """
    print(f"\nОпределитель матрицы A: {det_A:.2f}")
    print(f"Сумма диагональных элементов F: {sum_diag_F:.2f}")
    print(f"Условие (det(A) > sum(diag(F))): {condition_det}")
    """

    result = compute_expression(A, F, K, condition_det)
    
    """
    print("\nРезультат вычислений:")
    print(result)
    """

    plot_matrices(A, F, result)

if __name__ == "__main__":
    main()
