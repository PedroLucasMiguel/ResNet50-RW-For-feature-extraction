import torch
import torch.nn as nn
import resnet50_model as net

#Congela o calculo de gradiente das outras camadas
def __freeze_otrained_layers(model):
    for params in model.parameters():
        params.requires_grad = False
    return model

def create(n_classes: int, pre_trained: bool, train_just_fc: bool):
    #Configurando

    model = net.resnet50(pre_trained)

    #Verificando se sera necessário treinar as outras camadas
    if train_just_fc:
        model = __freeze_otrained_layers(model)

    #Criando a nova fully connected
    model.fc = nn.Linear(2048, n_classes)

    #Verificando disponibilidade CUDA
    if torch.cuda.is_available():
        print("CUDA disponivel. Modelo otimizado para uso de GPU")
        model = model.cuda()
    else:
        print("CUDA indisponível. Modelo otimizado para uso de CPU")

    return model
