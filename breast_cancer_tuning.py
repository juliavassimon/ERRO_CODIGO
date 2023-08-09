import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
#validação cruzada
from sklearn.model_selection import cross_val_score
from wrappers_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons): #função para criar a rede neural
    classificador = Sequential()
    #primeira camada oculta com 30 neuronios
    classificador.add(Dense(units=neurons, activation=activation, 
                            kernel_initializer=kernel_initializer, input_dim=30))
    #adicionar um droupout 
    classificador.add(Dropout(0.2))
    #segunda camada oculta
    classificador.add(Dense(units=neurons, activation=activation, 
                            kernel_initializer=kernel_initializer, input_dim=30))
    #adicionar um droupout 
    classificador.add(Dropout(0.2))
    #camada de saida
    classificador.add(Dense(units=1, activation='sigmoid'))

    #configuração/parametro para compilar
    classificador.compile(optimizer = optimizer, loss = loss, 
                          metrics = ['binary_accuracy'])
    return classificador

classificador= KerasClassifier(build_fn=criarRede())
parametros = {'batch_size': [10,30], 
              'epochs': [100],
              'optimizer': ['adam', 'sgd'],
              'loos': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16,8]}
grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
                               














