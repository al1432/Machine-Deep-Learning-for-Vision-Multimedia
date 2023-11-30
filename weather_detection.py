import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
import pandas as pd
import random
from keras.models import load_model
from imblearn.over_sampling import RandomOverSampler
import numpy as np


def stampa_info_dataset(train_images, val_images, test_images):
    total_images = len(train_images) + len(val_images) + len(test_images)
    print("Numero di immagini nel training set:", len(train_images))
    print("Numero di immagini nel validation set:", len(val_images))
    print("Numero di immagini nel test set:", len(test_images))
    print("Grandezza totale del dataset:", total_images)


def stampa_distribuzione_classi(labels, nome_set):
    plt.figure(figsize=(10, 6))
    sns.countplot(labels)
    plt.title(f'Distribuzione classi nel set: {nome_set}')
    plt.xlabel('Numero di campioni')
    plt.ylabel('Classe')
    plt.show()


def stampa_andamento_addestramento(history):
    plt.figure(figsize=(12, 6))

    # Grafico dell'accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Andamento dell\'Accuracy durante l\'addestramento')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy')
    plt.legend()

    # Grafico della loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Andamento della Loss durante l\'addestramento')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def build_complex_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Impostazioni
input_shape = (224, 224, 3)
batch_size = 32
epochs = 20
dataset_path = "C:\\Users\\alessandroamatori\\Desktop\\archive\\dataset"

# Creazione di una lista di percorsi delle immagini e relative etichette
images = []
labels = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        images.append(img_path)
        labels.append(category)

# Suddivisione del dataset in training, validation e test set
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42, stratify=train_labels
)

# Stampa informazioni sul dataset prima della data augmentation
stampa_info_dataset(train_images, val_images, test_images)
stampa_distribuzione_classi(train_labels, 'Training Set')
stampa_distribuzione_classi(val_labels, 'Validation Set')
stampa_distribuzione_classi(test_labels, 'Test Set')

# Stampa esempi per classe
classi_uniche = set(train_labels)
plt.figure(figsize=(15, 10))
for i, classe in enumerate(classi_uniche):
    plt.subplot(3, 4, i + 1)
    immagine_classe = [img for img, label in zip(train_images, train_labels) if label == classe][0]
    img = plt.imread(immagine_classe)
    plt.imshow(img)
    plt.title(f'Classe: {classe}')
    plt.axis('off')
plt.show()


# Data augmentation per il training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True

)

# Creazione di un DataFrame per il set di allenamento
train_df = pd.DataFrame({'filename': train_images, 'class': train_labels})

# Data augmentation per il training set
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Mostra un'immagine di esempio dopo la data augmentation appartenente a una classe casuale
plt.figure(figsize=(6, 6))

# Genera un batch di dati di esempio dopo la data augmentation
augmented_images, _ = train_generator.next()

# Seleziona casualmente un indice dal batch
random_index = random.randint(0, len(augmented_images) - 1)

# Visualizza l'immagine dopo la data augmentation appartenente a una classe casuale
plt.imshow(augmented_images[random_index])
plt.title('Immagine dopo la data augmentation')
plt.axis('off')
plt.show()

# Data normalization per il validation set
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Creazione di un DataFrame per il set di validazione
val_df = pd.DataFrame({'filename': val_images, 'class': val_labels})

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='class',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Data normalization per il test set
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Creazione di un DataFrame per il set di test
test_df = pd.DataFrame({'filename': test_images, 'class': test_labels})

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='class',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Creazione del modello CNN
complex_model = build_complex_model(input_shape, len(train_generator.class_indices))

# Definizione delle callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# Allenamento del modello con le callback
history_complex = complex_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[reduce_lr, early_stopping, checkpoint, tensorboard]
)

# Stampa andamento dell'addestramento
stampa_andamento_addestramento(history_complex)

# Valutazione del modello più complesso sul test set
test_metrics_complex = complex_model.evaluate(test_generator)
print(f"\nTest Set - Loss: {test_metrics_complex[0]}, Accuracy: {test_metrics_complex[1]}")

# Carica il modello salvato
saved_model = load_model("best_model.h5")

# Usa il modello su un'immagine di test a caso
random_test_image, random_test_label = test_generator.next()
prediction = saved_model.predict(random_test_image)

# Stampa l'output della predizione
print("Prediction:", prediction)
print("True Label:", random_test_label)
