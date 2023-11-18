import math
import os
from collections import Counter, defaultdict
from typing import Union, Tuple, Callable, List
import clip
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from Fairface_val import ff_val, Dotdict
# from debias_clip import DebiasCLIP, Adversary, ClipLike
from bfw_debias import DebiasCLIP, Adversary, ClipLike
from datasets import Dotdict, BFW

device = 'cuda'

def gen_prompts():
    raw_data = pd.read_csv('/home/elh33168/prompt_templates.csv')
    templates = raw_data["template"].tolist()
    concepts = raw_data["concept"].tolist()

    prompts = []
    for template in templates:
        new_list = []
        template = template.strip()
        if not template:
            continue
        new_list.extend(template.format(concept) for concept in concepts)
        prompts.append(new_list)
    return prompts

def get_similarity_scores(images_dl, model, prompts):

    folder_path = '/mnt/data4TBa/elh33168/data/BFW/bfw-cropped-aligned/'
    # folder_path = '/mnt/data4TBa/elh33168/data/FairFace/fairface-img-margin125-trainval/'
    # tensor_list = torch.empty((0,24))
    # tensor_list = []
    # images = pd.read_csv(val_labels)
    # images = images['file']
    # batch_scores = torch.empty(0,24).to(device)
    tensor_list = []
    print('beginning image embeddings')
    for batch in tqdm(images_dl, desc="Embedding images"):
        # count = 0
        for prompt in prompts:
            # for prompt_title in prompt_titles:      
                # print(image_features)
            
            these_scores = []
            for image in batch['file']:
                # print(prompt)
                image = folder_path + image
                image_input = preprocess(Image.open(image)).unsqueeze(0).to(device)
                text_inputs = clip.tokenize(prompt).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    text_features = model.encode_text(text_inputs)
                
                text_features /= text_features.norm(dim=-1, keepdim=True)
                image_features /= image_features.norm(dim=-1, keepdim=True)
        
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                # print(similarity)
                these_scores += similarity
            
            tensor_list += these_scores
            
    print('finished calculating similarity scores') 

    print(tensor_list)
    img_csv = "/home/elh33168/testing_image_embeddings.pt"

    torch.save(tensor_list, img_csv)

    return tensor_list

def normalized_discounted_KL(df: pd.DataFrame, num_prompt_tokens, top_n: int) -> dict:
    def KL_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    result_metrics = {f"ndkl_dem_par": 0.0}

    prompt_tokens = 24
    # if label count is 0, set it to 1 to avoid degeneracy
    desired_dist = {"dem_par": 1/num_prompt_tokens}
    
    for index, (_, row) in enumerate(top_n_scores.iterrows(), start=1):
        label = int(row["label"])
        top_n_label_counts[label] += 1
        for dist_name, dist in desired_dist.items():
            kl_div = KL_divergence(top_n_label_counts / index, dist)
            result_metrics[f"ndkl_{dist_name}"] += (
                kl_div / math.log2(index + 1))

    # print("label top_n", top_n_label_counts)

    Z = sum(1 / math.log2(i + 1)
            for i in range(1, top_n + 1))  # normalizing constant
    
    # print("normalizer: ", Z)
    # print("result metrics: ", result_metrics)

    for dist_name in result_metrics:
        result_metrics[dist_name] /= Z

    # original_out = sys.stdout
    # with open('racer_uneq.txt', 'a') as f:
    #     sys.stdout = f
    #     print("result_metrics: ", result_metrics)
    #     print(desired_dist)
    #     sys.stdout = original_out

    return result_metrics


def measure_bias(model, img_preproc, tokenizer, attribute):
    # val_labels = '/mnt/data4TBa/elh33168/data/BFW/small_bfw_val.csv'
    val_labels = '/home/elh33168/black_women.csv'
    val_img_path = '/mnt/data4TBa/elh33168/data/BFW/bfw-cropped-aligned/'

    # ds = ff_val(val_labels, val_img_path, iat_type=attribute)
    ds = BFW(val_labels, val_img_path, iat_type=attribute)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    prompts = gen_prompts()

    tensors = get_similarity_scores(dl, model, prompts)

    return tensors


class debiased_clip(nn.Module):
    def __init__(self, clip, adv):
        super(debiased_clip, self).__init__()
        #self.__init(clip, adv)
        self.clip = clip
        self.adv = adv
        self.classifier = nn.Linear(10, 10)

    def forward(self, x):
        x = self.clip(x)
        x = self.adv(x)
        return x

if __name__ == "__main__":
    device = 'cuda'
    # iat_type = 'race'
    # '''load saved models'''
    clip_model, preprocess = clip.load('ViT-B/16', device='cuda')
    tokenizer = clip_model.token_embedding


    # cfg_dic = Dotdict()
    # model = torch.load('/home/elh33168/saved_models/race_debias9.pt').to(device)
    # deb, preprocess, tokenizer = model.from_cfg(cfg_dic)
    # deb = deb.to(device)
    # deb.eval()

    # print(measure_bias(deb, preprocess, tokenizer, attribute='race'))
    # gender = measure_bias(deb, preprocess, tokenizer, attribute='gender')
    # race = measure_bias(deb, preprocess, tokenizer, attribute='race')

    measure_bias(clip_model, preprocess, tokenizer, attribute='race')

    # print('gender debiased ', gender)
    # print('race debiased ', race)