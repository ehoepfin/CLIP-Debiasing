import clip
import pandas as pd
import torch
import os

from PIL import Image
from Fairface_val import Dotdict 
from debias_clip import DebiasCLIP 


classes = ['successful', 'noncriminal', 'nonviolent', 'honest',
           'unsuccessful', 'caring', 'violent', 'dishonest', 'untrustworthy',
            'hardworking', 'trustworthy', 'clever', 'young']

# classes = ['smiling', 'violent', 'nonviolent', 'hardworking', 
#            'lazy', 'honest', 'dishonest', 'successful', 'unsuccessful', 'clever']

# directory = '/mnt/data4TBa/elh33168/data/BFW/bfw-cropped-aligned/'
directory = '/mnt/data4TBa/elh33168/data/FairFace/fairface-img-margin125-trainval/'
images = pd.read_csv('/home/elh33168/smiles.csv')
images = images['file']

# prompt = f"this is a photo of a {c} person"

# image = '/mnt/data4TBa/elh33168/data/FairFace/fairface-img-margin125-trainval/val/51.jpg'
# image = '/mnt/data4TBa/elh33168/data/BFW/bfw-cropped-aligned/white_females/n000898/0297_01.jpg'

'''now, lets see how CLIP classifies it'''

device = 'cuda'

my_clip, preprocess = clip.load('ViT-B/16', device)
# text_inputs = torch.cat([clip.tokenize(f"this is a {c} person") for c in classes]).to(device)
text_inputs = torch.cat([clip.tokenize(f"this person is {c}") for c in classes]).to(device)

tensor_list = []
for image in images:
    # image = directory + image
    image_input = preprocess(Image.open(f"{directory}{image}")).unsqueeze(0).to(device)
    # print(image_input)
    with torch.no_grad():
        image_features = my_clip.encode_image(image_input)
        # print(image_features)
        text_features = my_clip.encode_text(text_inputs)
        # print(text_features)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    tensor_list += similarity
    
# print(len(tensor_list))
concatenated_tensor = torch.stack(tensor_list, dim=0)
mean_tensor = torch.mean(concatenated_tensor, dim=0)
values, indices = mean_tensor.topk(13)

# Print the result
print("\nTop predictions from original CLIP:\n")
for value, index in zip(values, indices):
    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")

'''race debiased model'''
model = torch.load('/home/elh33168/saved_models/race_debias9.pt').to(device)
cfg_dic = Dotdict()
deb, deb_preprocess, tokenizer = model.from_cfg(cfg_dic)
deb = deb.to(device)
deb.eval()

# text_inputs = torch.cat([deb.tokenize(f"this person is {c}") for c in classes]).to(device)

tensor_list = []
for image in images:
    # image = directory + image
    image_input = deb_preprocess(Image.open(f"{directory}{image}")).unsqueeze(0).to(device)
    # print(image_input)
    with torch.no_grad():
        image_features = deb.encode_image(image_input)
        # print(image_features)
        text_features = deb.encode_text(text_inputs)
        # print(text_features)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    tensor_list += similarity
    
# print(len(tensor_list))
concatenated_tensor = torch.stack(tensor_list, dim=0)
mean_tensor = torch.mean(concatenated_tensor, dim=0)
values, indices = mean_tensor.topk(13)

# Print the result
print("\nTop predictions of race debiased CLIP:\n")
for value, index in zip(values, indices):
    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")


'''intersectionally debiased clip'''
int_model = torch.load('/home/elh33168/saved_models/eg_deb_9.pt').to(device)
int_deb, int_preprocess, int_tokenizer = int_model.from_cfg(cfg_dic)
int_deb = int_deb.to(device)

int_deb.eval()

# text_inputs = torch.cat([int_deb.tokenize(f"this person is {c}") for c in classes]).to(device)

tensor_list = []
for image in images:
    # image = directory + image
    image_input = int_preprocess(Image.open(f"{directory}{image}")).unsqueeze(0).to(device)
    # print(image_input)
    with torch.no_grad():
        image_features = int_deb.encode_image(image_input)
        # print(image_features)
        text_features = int_deb.encode_text(text_inputs)
        # print(text_features)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # print(similarity)
    tensor_list += similarity
    
concatenated_tensor = torch.stack(tensor_list, dim=0)
mean_tensor = torch.mean(concatenated_tensor, dim=0)
values, indices = mean_tensor.topk(13)
# Print the result
print("\nTop predictions from intersectionally debiased CLIP:\n")
for value, index in zip(values, indices):
    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")


# print("classifying photo of smiling face")
# image = '/mnt/data4TBa/elh33168/data/FairFace/fairface-img-margin125-trainval/val/4.jpg'
# image_input = int_preprocess(Image.open(image)).unsqueeze(0).to(device)
# # print(image_input)
# with torch.no_grad():
#     image_features = int_deb.encode_image(image_input)
#     # print(image_features)
#     text_features = int_deb.encode_text(text_inputs)
#     # print(text_features)

# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# values, indices = similarity[0].topk(5)

# print("\nTop predictions from intersectionally debiased CLIP:\n")
# for value, index in zip(values, indices):
#     print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")