"""
Разпознование цифр Mnist с помощью Keras (tensorflow)

https://github.com/meenuagarwal/MNIST-Classification-Using-Keras

"""
from typing import Tuple

import numpy as np
import cv2

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.src.utils import to_categorical

from keras.datasets import mnist

np.random.seed(123)


def create_model():
    """
    Создание простой модели
    :return:
    """
    new_model = Sequential()

    new_model.add(Flatten())
    new_model.add(Dense(128))
    new_model.add(Activation(tf.nn.relu))
    new_model.add(Dense(64))
    new_model.add(Activation(tf.nn.relu))
    new_model.add(Dense(10))
    new_model.add(Activation(tf.nn.sigmoid))

    return new_model


def create_model_conv2d():
    """
    Создание более сложной/тяжелой модели
    :return:
    """
    new_model = Sequential()

    new_model.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 1)))
    new_model.add(Activation(tf.nn.relu))
    new_model.add(Convolution2D(32, (3, 3)))
    new_model.add(Activation(tf.nn.relu))
    new_model.add(MaxPooling2D(pool_size=(2, 2)))
    new_model.add(Dropout(0.25))

    new_model.add(Flatten())
    new_model.add(Dense(128))
    new_model.add(Activation(tf.nn.relu))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(10))
    new_model.add(Activation(tf.nn.softmax))

    return new_model


def get_train_and_test_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Загрузка и подготовка датасета Mnist
    :return: (x_train, y_train), (x_test, y_test)
    """

    # загружаем датасет
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(dtype=float)
    x_test = x_test.astype(dtype=float)

    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def train_model(model_to_train, x_train: np.ndarray, y_train: np.ndarray, batch_size=32, epochs=10, verbose=1) -> None:
    """
    Обучение модели
    :param model_to_train: Модель
    :param x_train:
    :param y_train:
    :param batch_size:
    :param epochs:
    :param verbose:
    :return:
    """

    # выбираем оптимизатор
    optimizer = tf.keras.optimizers.SGD()

    # функция потерь
    loss = tf.keras.losses.CategoricalCrossentropy()

    # подготовка модели и задание метрик для обучения
    model_to_train.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    # обучение модели
    model_to_train.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)


def test_model(model_to_test, x_test: np.ndarray, y_test: np.ndarray, show_image=True) -> None:
    """
    Проверка модели на тестовой выборке
    :param model_to_test:
    :param x_test:
    :param y_test:
    :param show_image: Показывать картинку
    :return:
    """

    # выбираем несколько картинок случайным образом
    for i in np.random.choice(np.arange(0, len(y_test)), size=(20,)):
        # распознавание
        probs = model_to_test.predict(x_test[np.newaxis, i], verbose=0)
        # Выбираем с макс. вероятностью
        prediction = probs.argmax(axis=1)

        print(f"{i}, Actual digit is {y_test[i].argmax()}, predicted {prediction[0]}")

        if show_image:
            image = (x_test[i] * 255).reshape((28, 28)).astype("uint8")

            cv2.imshow("Digit", image)
            cv2.waitKey(0)


def model_1() -> None:
    """
    Test simple model
    :return:
    """
    print("Test simple model")
    model = create_model()

    (x_train, y_train), (x_test, y_test) = get_train_and_test_data()

    train_model(model_to_train=model, x_train=x_train, y_train=y_train)
    test_model(model_to_test=model, x_test=x_test, y_test=y_test, show_image=False)


def model_2() -> None:
    """
    Test model with 'conv2d' layers
    :return:
    """
    print("Test model with conv2d layers")

    model = create_model_conv2d()

    (x_train, y_train), (x_test, y_test) = get_train_and_test_data()

    train_model(model_to_train=model, x_train=x_train, y_train=y_train)
    test_model(model_to_test=model, x_test=x_test, y_test=y_test, show_image=False)


if __name__ == '__main__':
    model_1()
    model_2()
