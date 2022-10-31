from typing import Any
from numpy import array, zeros, hstack, tanh, sin, insert, subtract, diag, outer, round
from numpy.random import random_sample

from matplotlib import pyplot
from clip_dataset import clipDataset
import pandas

data = None
try:
    data = pandas.read_csv("./res/clipped.csv")
except:
    data = clipDataset()

class Network:
    # pesos da rede
    weights: list[list[Any]]
    out_weights: list[list[Any]]
    # bias da rede
    bias: float
    # taxa de aprendizado da rede
    eta: float
    
    @staticmethod
    def start(entry: int, hidden: int, out: int, bias: float, eta: float):
        """inicia a rede para poder ser treinada

        Args:
            prototype (list[int]): uma lista ordenada aonde cada posição representa o tamanho de uma camada.
        """
        Network.weights = random_sample((hidden, entry+1))
        Network.out_weights = random_sample((out, hidden+1))
        Network.bias = bias
        Network.eta = eta
    
    @staticmethod
    def feedForward(entry: list[Any]):
        """executa a classificação da entrada

        Args:
            entry (list[Any]): vetor de entradas

        Returns (list[Any]): retorna o vetor de saida
        """
        
        entry = hstack([Network.bias, entry])
        w_out = tanh(Network.weights.dot(entry))
        wb_out = insert(w_out, 0, Network.bias)
        return tanh(Network.out_weights.dot(wb_out)), wb_out, entry
    
    @staticmethod
    def backPropagation(error, entry, out, wb_out):
        delta2 = diag(error).dot((1 - out*out))
        vdelta2 = (Network.out_weights.transpose()).dot(delta2)
        delta1 = diag(1 - wb_out*wb_out).dot(vdelta2)
        
        Network.weights = Network.weights + Network.eta*(outer(delta1[1:], entry))
        Network.out_weights = Network.out_weights + Network.eta*(outer(delta2, wb_out))
    
    @staticmethod
    def fit(batchs: list, validate: list, epochs: int):
        """executa o treinamento da rede

        Args:
            batchs (list): pacotes de treino
            validate (list): saidas esperadas
            epochs (int): epocas de treinamento
        """
        errors = zeros(epochs)
        e_errors = zeros(len(batchs))
        min_e = 1
        
        for epoch in range(epochs):
            e = 0
            for expect, batch in zip(validate, batchs):
                out, wb_out, entry = Network.feedForward(batch)
                
                error = subtract(expect, out)
                e_errors[e] = (error.transpose().dot(error))/2 
                
                Network.backPropagation(error, entry, out, wb_out)
                e+=1
            e_mean = e_errors.mean()
            errors[epoch] = e_mean
            if e_mean < min_e: min_e = e_mean
            print("Epoch "+ str(epoch)+ " finalized! with "+str(e_mean)+" mean error. Best run "+str(min_e)+" mean error", end="\r")
        print("Epoch "+ str(epoch)+ " finalized! with "+str(e_mean)+" mean error. Best run "+str(min_e)+" mean error")

        pyplot.xlabel("Épocas")
        pyplot.ylabel("Erro Médio")
        pyplot.plot(errors, color='b')
        pyplot.plot(errors)
        pyplot.show()
        
    @staticmethod
    def test(batchs: list, validate: list):
        t_errors = zeros(len(batchs))

        e = 0
        for expect, batch in zip(validate, batchs):
            out, *_ = Network.feedForward(batch)
            
            error = subtract(expect, out)
            t_errors[e] = (error.transpose().dot(error))/2 
            e+=1

        print(t_errors)
        print("Saida dos testes (0 é bom): ", round(t_errors)-validate)
        
if __name__ == "__main__":
    # "Action" = 0
    # "Sports" = 1
    Network.start(
        entry=11, #dinamico
        hidden=8,
        out=1,
        bias=1.2,
        eta=0.2
    )
    
    validate = data.pop("Genre").to_list()
    batchs = data[data.columns[1:]].to_numpy()
    
    Network.fit(
        batchs,
        validate,
        epochs=2000
    )
    
    Network.test(
        batchs,
        validate
    )