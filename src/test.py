import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
from numpy import asarray
from numpy import savetxt
import fine_tuned_models as ftm

features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output
    return hook

def get_features_batch(name):
    def hook(model, input, output):
        relu = nn.ReLU(inplace=True)
        features[name] = relu(output)
    return hook

model = ftm.custom_resnet50(2, True, False)
model.load_state_dict(torch.load('model_chekpoint/best_model_41_f1=0.9018.pt'))
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
print(model)
model.maxpool.register_forward_hook(get_features('max_pooling2d_1'))
model.layer1[0].relu.register_forward_hook(get_features('activation_4_relu'))
model.layer4[2].bn2.register_forward_hook(get_features_batch('activation_48_relu'))
model.layer4[2].relu.register_forward_hook(get_features('activation_49_relu'))
model.avgpool.register_forward_hook(get_features('avg_pool'))

image = PIL.Image.open('Covid (185).png')
image = transform(image)
image = image.unsqueeze(0)
image = image.cuda()

model.eval()
model(image)
print(model)

print(features['max_pooling2d_1'].shape)
print(features['activation_4_relu'].shape)
print(features['activation_48_relu'].shape)
print(features['activation_49_relu'].shape)
print(features['avg_pool'].shape)






