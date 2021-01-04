import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL
from PIL import Image
import Custom_ResNet50.fine_tunned_model as ftm
from Arfflib import Arfflib as arff

test = {}
max_pooling2d_1_f = arff("max_pooling2d_1", 193600)
activation_4_relu = None
activation_48_relu = None
activation_49_relu = None
avg_pool = None
results = None

def get_features(name):
    def hook(model, input, output):
        aux_array = output.cpu().detach().numpy()
        aux_shape = aux_array.shape
        aux_array = aux_array.reshape(aux_shape[1] * aux_shape[2], aux_shape[3])
        aux_array = aux_array.flatten()
        test["test"] = aux_array
    return hook

model = ftm.create(2, False, True)
model.load_state_dict(torch.load('checkpoints/best_model_41_f1=0.9018.pt'))

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

model.maxpool.register_forward_hook(get_features('max_pooling2d_1'))

image = PIL.Image.open('Healthy (545).png')
image = transform(image)
image = image.unsqueeze(0)
image = image.cuda()

model.eval()
out = model(image)
print(test["test"])

if out[0][0] > out[0][1]:
    max_pooling2d_1_f.append(test["test"], 0)
else:
    max_pooling2d_1_f.append(test["test"], 1)

image = PIL.Image.open('Covid (185).png')
image = transform(image)
image = image.unsqueeze(0)
image = image.cuda()

model.eval()
out = model(image)
print(test["test"])

if out[0][0] > out[0][1]:
    max_pooling2d_1_f.append(test["test"], 0)
else:
    max_pooling2d_1_f.append(test["test"], 1)

max_pooling2d_1_f.close()






