import numpy as np
import matplotlib.pyplot as plt

def generate_food_data(n_samples=300):
    np.random.seed(42)
    
    food_params = {
        'Vegetables': {'sweetness': (2, 1.0), 'crunchiness': (8, 1.5)},   # not sweet, crunchy
        'Fruits':     {'sweetness': (7, 1.5), 'crunchiness': (5, 1.2)},   # sweet, moderately crunchy
        'Protein':    {'sweetness': (1, 0.5), 'crunchiness': (3, 1.0)},   # not sweet, not crunchy
    }
    
    X = []
    y = []
    labels = []
    
    for i, (food_name, params) in enumerate(food_params.items()):
        for _ in range(n_samples // len(food_params)):
            sweetness = np.random.normal(params['sweetness'][0], params['sweetness'][1])
            crunchiness = np.random.normal(params['crunchiness'][0], params['crunchiness'][1])
            
            sweetness = np.clip(sweetness, 0, 10)
            crunchiness = np.clip(crunchiness, 0, 10)
            
            X.append([sweetness, crunchiness])
            y.append(i)
            labels.append(food_name)
    
    return np.array(X), np.array(y), list(food_params.keys())

class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            
            prediction = np.bincount(nearest_labels).argmax()
            predictions.append(prediction)
        return np.array(predictions)

def split_data(X, y, test_size=0.3):
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def plot_food_results(X, y, class_names, predictions=None, title="Классификатор"):
    plt.figure(figsize=(12, 8))
    
    colors = ['green', 'red', 'brown'] 

    if predictions is not None:
        correct = y == predictions
        incorrect = ~correct
        
        for i in range(len(class_names)):
            class_correct = correct & (y == i)
            if np.any(class_correct):
                plt.scatter(X[class_correct, 0], X[class_correct, 1], 
                           c=colors[i], s=60, alpha=0.7, label=f'{class_names[i]} (correct)')
        
        if np.any(incorrect):
            plt.scatter(X[incorrect, 0], X[incorrect, 1], 
                       c='black', s=150, marker='x', linewidth=2, 
                       label='Ошибка')
        
        accuracy = np.sum(correct) / len(y)
        title += f" (Точность: {accuracy:.3f})"
    else:
        for i in range(len(class_names)):
            class_mask = y == i
            plt.scatter(X[class_mask, 0], X[class_mask, 1], 
                       c=colors[i], s=50, alpha=0.7, label=class_names[i])
    
    plt.xlabel('Сладость (0-10)', fontsize=12)
    plt.ylabel('Хруст (0-10)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()

def main_experiment():
    X, y, class_names = generate_food_data(n_samples=300)
    
    print("Характеристики:")
    for i, name in enumerate(class_names):
        class_data = X[y == i]
        print(f"{name}: сладость {class_data[:, 0].mean():.1f} ± {class_data[:, 0].std():.1f}, "
              f"хруст {class_data[:, 1].mean():.1f} ± {class_data[:, 1].std():.1f}")
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
    print(f"\nНабор обучающих данных: {len(X_train)}")
    print(f"Контрольный набор данных: {len(X_test)}")
    
    knn = SimpleKNN(k=5)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    print(f"\nТочность классификации: {accuracy:.4f}")
    
    plot_food_results(X, y, class_names, title="Набор данных")
    plot_food_results(X_test, y_test, class_names, y_pred, title="Результаты классификации")

def test_specific_foods():
    X, y, class_names = generate_food_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    knn = SimpleKNN(k=5)
    knn.fit(X_train, y_train)
    
    specific_foods = {
        'Carrot':     [3.0, 9.0],  
        'Cucumber':   [1.0, 8.5],  
        'Apple':      [7.5, 6.0],  
        'Banana':     [6.0, 2.0],  
        'Bacon':      [1.5, 4.0],  
        'Cheese':     [1.0, 2.0],  
        'Fish':       [1.0, 3.0],  
        'Watermelon': [8.5, 3.0],  
        'Cabbage':    [1.5, 7.0],   
    }
    
    test_examples = np.array(list(specific_foods.values()))
    food_names = list(specific_foods.keys())
    
    predictions = knn.predict(test_examples)
    
    for i, (food_name, example, pred) in enumerate(zip(food_names, test_examples, predictions)):
        print(f"{food_name:10} | {example[0]:9.1f} | {example[1]:11.1f} | {class_names[pred]}")
    
    plt.figure(figsize=(12, 8))
    
    colors = ['green', 'red', 'brown']
    for i in range(len(class_names)):
        class_mask = y_train == i
        plt.scatter(X_train[class_mask, 0], X_train[class_mask, 1], 
                   c=colors[i], alpha=0.1, s=30, label=f'{class_names[i]} (обучение)')
    
    for i, (food_name, example, pred) in enumerate(zip(food_names, test_examples, predictions)):
        plt.scatter(example[0], example[1], 
                   c=colors[pred], s=150, marker='o', edgecolor='black', linewidth=2)
        plt.annotate(food_name, (example[0] + 0.1, example[1] + 0.1), 
                    fontsize=9, fontweight='bold')
    
    plt.xlabel('Сладость (0-10)', fontsize=12)
    plt.ylabel('Хруст (0-10)', fontsize=12)
    plt.title('Классификатор', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()

def experiment_different_k():
    X, y, class_names = generate_food_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    k_values = [1, 3, 5, 7, 9, 11, 15]
    accuracies = []
    
    for k in k_values:
        knn = SimpleKNN(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        accuracies.append(accuracy)
        print(f"k = {k:2d}: точность = {accuracy:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('k значение', fontsize=12)
    plt.ylabel('Точность', fontsize=12)
    plt.title('Зависимость точности от k-значения', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.show()

if __name__ == "__main__":
    main_experiment()
    test_specific_foods()
    experiment_different_k()
