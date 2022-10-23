import torch
import numpy as np
import clip

device='cpu'
model, preprocess = clip.load("ViT-B/32", device=device)

image=torch.randn((3,3,224,224))
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

image_features = model.encode_image(image)
text_features = model.encode_text(text)
a=1