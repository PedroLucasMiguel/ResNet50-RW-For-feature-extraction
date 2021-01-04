import torch
import torch.nn as nn
import models.feature_extraction_resnet50 as n
import models.normal_Resnet50 as fe

#Congela o calculo de gradiente das outras camadas
def __freeze_trained_layers(model):
    for params in model.parameters():
        params.requires_grad = False
    return model

def create(n_classes: int, pre_trained: bool, train_just_fc: bool, is_fe: bool):
    #Configurando

    #Verificando o tipo de rede
    if is_fe:
        model = fe.resnet50(pre_trained)
    else:
        model = n.resnet50(pre_trained)

    #Verificando se sera necessário treinar as outras camadas
    if train_just_fc:
        model = __freeze_trained_layers(model)

    #Criando a nova fully connected
    model.fc = nn.Linear(2048, n_classes)

    #Verificando disponibilidade CUDA
    if torch.cuda.is_available():
        print("CUDA disponivel. Modelo otimizado para uso de GPU")
        model = model.cuda()
    else:
        print("CUDA indisponível. Modelo otimizado para uso de CPU")
