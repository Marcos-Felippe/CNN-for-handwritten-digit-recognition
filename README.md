## Convolutional Neural Network (CNN) para Classificação de Dígitos do MNIST

Este repositório contém um exemplo de implementação de uma Rede Neural Convolucional (CNN) utilizando TensorFlow/Keras para classificação de dígitos escritos à mão do dataset MNIST.
O código realiza todo o fluxo: carregamento dos dados, pré-processamento, construção do modelo, treinamento e avaliação.

### Características do Projeto

- Utiliza o dataset MNIST (60.000 imagens de treino + 10.000 de teste).
- Modelo baseado em camadas Conv2D, MaxPooling2D, Dropout e Dense.
- Treinamento por 20 épocas usando Adam como otimizador.
- Salva o modelo final no arquivo modelo1.h5.
- Avalia acurácia e perda no conjunto de teste.

### Estrutura do Código

O script cnn_example.py contém:

#### 1. Instalação e Importação das Bibliotecas
- As bibliotecas principais são TensorFlow/Keras e Matplotlib.

#### 2. Carregamento e Pré-processamento dos Dados
Inclui:
- Normalização das imagens (0–1)
- Redimensionamento para (28, 28, 1)
- Conversão das labels para one-hot encoding

#### 3. Construção da CNN
A rede contém:
- Conv2D (30 filtros, kernel 5x5, ReLU)
- MaxPooling2D (2x2)
- Conv2D (15 filtros, kernel 3x3, ReLU)
- MaxPooling2D (2x2)
- Dropout (0.2)
- Flatten
- Dense (128 → 64 → 32 neurônios, ReLU)
- Dense final (10 neurônios, softmax)

#### 4. Compilação e Treinamento
- Loss: categorical_crossentropy
- Métrica: accuracy
- Otimizador: Adam
- Épocas: 20

#### 5. Avaliação
O código imprime:
- Acurácia (%)
- Perda (%)

### Resultados
Com este exemplo sem alterações e melhorias já é possivel chegar a uma acurácia e perda muito boas, de 99,29% e 2,58% respectivamente como pode ser observado abaixo:
<br/><br/>
<img width="757" height="383" alt="Captura de tela 2025-12-01 221554" src="https://github.com/user-attachments/assets/1b1c07ce-680e-4c93-8224-3980162a7188" />
