import os
import pandas as pd
import numpy as np
from PIL import Image

mega_df = pd.read_csv(
    "/mnt/data4TBa/elh33168/data/BFW/bfw-v0.1.5-datatable.csv")

'''fix dataframe by vertically combining them
first, make 2 dataframes of each part'''

p1 = mega_df['p1'].tolist()
p2 = mega_df['p2'].tolist()

#id1 = mega_df['id1'].tolist()
#id2 = mega_df['id2'].tolist()

att1 = mega_df['att1'].tolist()
att2 = mega_df['att2'].tolist()

a1 = mega_df['a1'].tolist()
a2 = mega_df['a2'].tolist()

g1 = mega_df['g1'].tolist()
g2 = mega_df['g2'].tolist()

e1 = mega_df['e1'].tolist()
e2 = mega_df['e2'].tolist()

df1 = pd.DataFrame()
df2 = pd.DataFrame()

df1['file'] = p1
df2['file'] = p2

df1['gender'] = g1
df2['gender'] = g2

df1['race'] = e1
df2['race'] = e2

df1['ethnic_gender'] = a1
df2['ethnic_gender'] = a2

new_val = pd.concat([df1, df2], axis=0)

new_val = new_val.reset_index()

# small_val = new_val.sample(n=2000)
n = 50
# #select 10000 rows from each class
# am_df = new_val.loc[new_val['ethnic_gender'] == 'AM']
# am = am_df.sample(n)

# af_df = new_val.loc[new_val['ethnic_gender'] == 'AF']
# af = af_df.sample(n)

# bm_df = new_val.loc[new_val['ethnic_gender'] == 'BM']
# bm = bm_df.sample(n)

# bf_df = new_val.loc[new_val['ethnic_gender'] == 'BF']
# bf = bf_df.sample(n)

# im_df = new_val.loc[new_val['ethnic_gender'] == 'IM']
# im = im_df.sample(n)

# if_df = new_val.loc[new_val['ethnic_gender'] == 'IF']
# i = if_df.sample(n)

# wm_df = new_val.loc[new_val['ethnic_gender'] == 'WM']
# wm = wm_df.sample(n)

# wf_df = new_val.loc[new_val['ethnic_gender'] == 'WF']
# wf = wf_df.sample(n)

# new_small = pd.concat([am, af, bm, bf, im, i, wm, wf])
# new_small = new_small.sample(frac=1)

new_small = new_val.sample(50)

# print(new_small.sample)
# print(new_small.sample)

'''save validation to .csv'''
# new_val.to_csv(
#     '/mnt/data4TBa/elh33168/data/BFW/exp_bfw_train.csv', index=False)

new_small.to_csv('/home/elh33168/small_sample.csv', index=False)
# new_val = im
# new_val.to_csv('/home/elh33168/indian_males.csv')
