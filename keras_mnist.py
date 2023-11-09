"""
Разпознование цифр Mnist с помощью Keras (tensorflow)
"""
from typing import Tuple

import numpy as np
import cv2

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.src.utils import to_categorical

from keras.datasets import mnist

np.random.seed(123)


def create_model():
    new_model = Sequential()

    new_model.add(Flatten())
    new_model.add(Dense(128))
    new_model.add(Activation(tf.nn.relu))
    new_model.add(Dense(64))
    new_model.add(Activation(tf.nn.relu))
    new_model.add(Dense(10))
    new_model.add(Activation(tf.nn.sigmoid))

    return new_model


def get_train_and_test_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Загрузка и подготовка датасета Mnist
    :return: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    # X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

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
    Проверка модели на тестовой выбборке
    :param model_to_test:
    :param x_test:
    :param y_test:
    :param show_image: Показывать картинку
    :return:
    """

    # выбираем несколько картиток случайным образом
    for i in np.random.choice(np.arange(0, len(y_test)), size=(20,)):
        # распознование
        probs = model_to_test.predict(x_test[np.newaxis, i], verbose=0)
        # выбираем с макс. вероятностью
        prediction = probs.argmax(axis=1)

        print(f"{i}, Actual digit is {y_test[i].argmax()}, predicted {prediction[0]}")

        if show_image:
            image = (x_test[i] * 255).reshape((28, 28)).astype("uint8")

            cv2.imshow("Digit", image)
            cv2.waitKey(0)


def model_1():
    model = create_model()

    (X_train, y_train), (X_test, y_test) = get_train_and_test_data()

    train_model(model_to_train=model, x_train=X_train, y_train=y_train)
    test_model(model_to_test=model, x_test=X_test, y_test=y_test, show_image=False)


model_1()
