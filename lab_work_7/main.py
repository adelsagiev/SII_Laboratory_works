import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def generate_food_data(n_samples=900):
    np.random.seed(42)
    
    food_params = {
        'Vegetables': {'sweetness': (2, 1.0), 'crunchiness': (8, 1.5)},
        'Fruits': {'sweetness': (7, 1.5), 'crunchiness': (5, 1.2)},
        'Protein': {'sweetness': (1, 0.5), 'crunchiness': (3, 1.0)},
    }
    
    X = []
    y = []
    class_names = list(food_params.keys())
    
    for class_id, (food_name, params) in enumerate(food_params.items()):
        for _ in range(n_samples // len(food_params)):
            sweetness = np.random.normal(params['sweetness'][0], params['sweetness'][1])
            crunchiness = np.random.normal(params['crunchiness'][0], params['crunchiness'][1])
            
            sweetness = np.clip(sweetness, 0, 10)
            crunchiness = np.clip(crunchiness, 0, 10)
            
            X.append([sweetness, crunchiness])
            y.append(class_id)
    
    return np.array(X), np.array(y), class_names

X, y, class_names = generate_food_data(n_samples=900)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Classes: {len(class_names)}")
print(f"Class names: {class_names}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train_cat = to_categorical(y_train, len(class_names))
y_test_cat = to_categorical(y_test, len(class_names))

input_size = X_train.shape[1]

model = Sequential()

model.add(Dense(64, input_dim=input_size, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=32,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

y_pred = np.argmax(model.predict(X_test), axis=1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

axes[2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[2].set_title('Confusion Matrix')
axes[2].set_xticks(range(len(class_names)))
axes[2].set_yticks(range(len(class_names)))
axes[2].set_xticklabels(class_names)
axes[2].set_yticklabels(class_names)
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True')

for i in range(len(class_names)):
    for j in range(len(class_names)):
        axes[2].text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")

plt.tight_layout()
plt.show()

test_foods = {
    'Carrot': [3.0, 9.0],
    'Cucumber': [1.0, 8.5],
    'Apple': [7.5, 6.0],
    'Banana': [6.0, 2.0],
    'Bacon': [1.5, 4.0],
    'Cheese': [1.0, 2.0],
}

for food_name, features in test_foods.items():
    features_norm = scaler.transform([features])
    prediction = model.predict(features_norm, verbose=0)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    
    print(f"{food_name:10} | Sweetness: {features[0]:.1f}, Crunchiness: {features[1]:.1f} | "
          f"Class: {class_names[class_id]} (confidence: {confidence:.2f})")

model.save('food_classifier_model.h5')
print("Model saved as 'food_classifier_model.h5'")
