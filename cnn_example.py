# -*- coding: utf-8 -*-
"""cnn_example.ipynb
## Etapa 1: Instando o TensorFlow e Keras
"""

!pip install tensorflow
!pip install keras

"""## Etapa 2: Importando as bibliotecas"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import to_categorical

from matplotlib import pyplot as plt

"""## Etapa 3: Pré-processamento

### Carregando a base de dados MNIST
"""

# Carregando a base de dados
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Pegando os dados de treino e teste do dataset

def load_data():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # Tornando os valores dos pixels entre 1. e 0.
    x_train = x_train / 255
    x_test = x_test / 255

    # Categorizando cada valor de label de acordo com um valor one hot.
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()

x_train[0]

x_train.shape

x_test.shape

plt.imshow(x_test[2])

"""## Etapa 4: Construindo a Rede Neural Convolucional

### Definindo o modelo
"""

#model = tf.keras.models.Sequential()
filename = 'modelo1.h5'

# pegando a quantidade de classes do dataset
num_classes = y_test.shape[1]

#Criando o modelo
model = Sequential()

"""### Adicionado a primeira camada de convolução

Hyper-parâmetros da camada de convolução:
- filters (filtros): 32
- kernel_size (tamanho do kernel): 3
- padding (preenchimento): same
- função de ativação: relu
- input_shape (camada de entrada): (32, 32, 3)

"""

model.add(Conv2D(filters=30, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))

"""### Adicionando a camada de max-pooling e a segunda camada de convolução

Hyper-parâmetros da camada de max-pooling:
- pool_size: 2
- strides: 2
- padding: valid

Hyper-parâmetros da camada de convolução:
- filters (filtros): 32
- kernel_size (tamanho do kernel):3
- padding (preenchimento): same
- função de ativação: relu

"""

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(filters=15, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))

"""### Adicionando a segunda camada de max-pooling

Hyper-parâmetros da camada de convolução:

    filters: 64
    kernel_size:3
    padding: same
    activation: relu

"""

model.add(MaxPooling2D(pool_size=(2, 2)))

"""###  Adicionando uma camada de Dropout

Hyper-parâmetros da camada de convolução:

    filters: 64
    kernel_size:3
    padding: same
    activation: relu

"""

model.add(Dropout(0.2))

"""### Adicionando a camada de flattening"""

model.add(Flatten())

"""### Adicionando três camadas densas (fully-connected)

Hyper-parâmetros da 1° camada densa:
- units/neurônios: 128
- função de ativação: relu

Hyper-parâmetros da 2° camada densa:
- units/neurônios: 64
- função de ativação: relu

Hyper-parâmetros da 3° camada densa:
- units/neurônios: 32
- função de ativação: relu
"""

model.add(Dense(units = 128, activation='relu'))

model.add(Dense(units = 64, activation='relu'))

model.add(Dense(units = 32, activation='relu'))

"""### Adicionando a camada de saída

Hyper-parâmetros da camada de saída:

 - units/neurônios: 10 (número de classes)
 - activation: softmax

"""

model.add(Dense(units=num_classes, activation='softmax'))

model.summary()

"""### Compilando o modelo

#### sparse_categorical_accuracy
"""

# 0 0 0 1 0 0 0 0 0 0
y_test[0]

#model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

"""### Treinando o modelo"""

model.fit(x_train, y_train, epochs=20)

"""### Avaliando o modelo"""

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Acurácia de Teste: {}".format(test_accuracy))
print("\nAcurácia em porcentagem: %.2f%%" % (test_accuracy*100))

print("Perda de Teste: {}".format(test_loss))
print("\nPerda em porcentagem: %.2f%%" % (test_loss*100))
