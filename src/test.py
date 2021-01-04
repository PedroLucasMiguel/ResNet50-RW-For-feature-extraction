import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL
from PIL import Image
import rs50_models.fine_tunned_model as ftm

model = ftm.create(2, False, True)
model.load_state_dict(torch.load('checkpoints/best_model_41_f1=0.9018.pt'))

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

print(model)

image = PIL.Image.open('Covid (185).png')
image = transform(image)
image = image.unsqueeze(0)
image = image.cuda()

model.eval()
out = model(image)
print(out[0])






