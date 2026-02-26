import os
import random
import numpy as np
import keras
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.applications import VGG16

root_dir = './eagles_chickens_organized/'

train_path = os.path.join(root_dir, 'train')

categories = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])

num_classes = len(categories)
print(f"Class detected: {categories}")


train_split_ratio, val_split_ratio = 0.7, 0.15

def get_image(path):
    img = image.load_img(path, target_size=(224, 224)) 
    x = image.img_to_array(img) 
    x = np.expand_dims(x, axis=0) 
    x = preprocess_input(x) 
    return img, x

x_train_list, y_train_list = [], []
x_val_list, y_val_list = [], []
x_test_list, y_test_list = [], []

category_to_int = {category: i for i, category in enumerate(categories)}

for c_name in categories:
    train_category_path = os.path.join(root_dir, 'train', c_name)
    if os.path.exists(train_category_path):
        train_images = [os.path.join(train_category_path, f) for f in os.listdir(train_category_path)
                        if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg', '.jfif']]
        for img_path in train_images:
            img, x = get_image(img_path)
            if x is not None:
                x_train_list.append(np.array(x[0]))
                y_train_list.append(category_to_int[c_name])

    
    val_category_path = os.path.join(root_dir, 'validation', c_name)
    if os.path.exists(val_category_path):
        val_images = [os.path.join(val_category_path, f) for f in os.listdir(val_category_path)
                      if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg', '.jfif']]
        for img_path in val_images:
            img, x = get_image(img_path)
            if x is not None:
                x_val_list.append(np.array(x[0]))
                y_val_list.append(category_to_int[c_name])


    test_category_path = os.path.join(root_dir, 'test', c_name)
    if os.path.exists(test_category_path):
        test_images = [os.path.join(test_category_path, f) for f in os.listdir(test_category_path)
                       if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg', '.jfif']]
        for img_path in test_images:
            img, x = get_image(img_path)
            if x is not None:
                x_test_list.append(np.array(x[0]))
                y_test_list.append(category_to_int[c_name])

x_train, y_train = np.array(x_train_list), np.array(y_train_list)
x_val, y_val = np.array(x_val_list), np.array(y_val_list)
x_test, y_test = np.array(x_test_list), np.array(y_test_list)

x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"Image uploading completed for {num_classes} categorias: {categories}.")
print(f"Training/validation/testing division: {len(x_train)}, {len(x_val)}, {len(x_test)}")
print(f"Training data format (images): {x_train.shape}")
print(f"Training label format: {y_train.shape}")

num_display_images = 8
if len(x_test) >= num_display_images:
    display_indices = random.sample(range(len(x_test)), num_display_images)
    display_images = [x_test[i] for i in display_indices]

    concat_image = np.concatenate([(img * 255).astype(np.uint8) for img in display_images], axis=1)

    plt.figure(figsize=(16, 4))
    plt.imshow(concat_image)
    plt.axis('off') # Remove os eixos para uma visualização mais limpa
    plt.title(f'Sample images from Dataset: {', '.join(categories)}')
    plt.show()

model_scratch = Sequential()
print("Input dimensions for the zero network: ", x_train.shape[1:])

model_scratch.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model_scratch.add(Activation('relu')) 
model_scratch.add(MaxPooling2D(pool_size=(2, 2)))

model_scratch.add(Conv2D(32, (3, 3)))
model_scratch.add(Activation('relu'))
model_scratch.add(MaxPooling2D(pool_size=(2, 2)))

model_scratch.add(Conv2D(64, (3, 3)))
model_scratch.add(Activation('relu'))
model_scratch.add(MaxPooling2D(pool_size=(2, 2)))

model_scratch.add(Flatten()) 
model_scratch.add(Dense(64)) 
model_scratch.add(Activation('relu'))
model_scratch.add(Dropout(0.5)) 
model_scratch.add(Dense(num_classes)) 
model_scratch.add(Activation('softmax'))

model_scratch.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

model_scratch.summary()

history_scratch = model_scratch.fit(x_train, y_train,
                                    batch_size=32,
                                    epochs=10,
                                    validation_data=(x_val, y_val),
                                    verbose=1)

scores_scratch = model_scratch.evaluate(x_test, y_test, verbose=1)
print(f"Zero network accuracy in testing: {scores_scratch[1]*100:.2f}%") 



# Applying Transfer Learning with VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=x_train.shape[1:])

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x) 
x = Dense(256, activation='relu')(x) 
x = Dropout(0.5)(x) 
predictions = Dense(num_classes, activation='softmax')(x)

model_tl = Model(inputs=base_model.input, outputs=predictions)

model_tl.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

model_tl.summary()

history_tl = model_tl.fit(x_train, y_train,
                          batch_size=32,
                          epochs=10,
                          validation_data=(x_val, y_val),
                          verbose=1)

scores_tl = model_tl.evaluate(x_test, y_test, verbose=1)
print(f"Accuracy of the Transfer Learning model in the test: {scores_tl[1]*100:.2f}%")


#Comparison of Results

plt.figure(figsize=(12, 4))


#Accuracy Chart
plt.subplot(1, 2, 1)
plt.plot(history_scratch.history['accuracy'], label='Acurácia Treino (Scratch)')
plt.plot(history_scratch.history['val_accuracy'], label='Acurácia Validação (Scratch)')
plt.plot(history_tl.history['accuracy'], label='Acurácia Treino (Transfer Learning)')
plt.plot(history_tl.history['val_accuracy'], label='Acurácia Validação (Transfer Learning)')
plt.title('Comparação de Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()


#Loss chart
plt.subplot(1, 2, 2)
plt.plot(history_scratch.history['loss'], label='Perda Treino (Scratch)')
plt.plot(history_scratch.history['val_loss'], label='Perda Validação (Scratch)')
plt.plot(history_tl.history['loss'], label='Perda Treino (Transfer Learning)')
plt.plot(history_tl.history['val_loss'], label='Perda Validação (Transfer Learning)')
plt.title('Comparação de Perda')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.show()
