import numpy as np
import pandas as pd
from PIL import Image
import os, time

t0=time.time()
labels_df = pd.read_csv('data/train_labels.csv', dtype={'id':str})
classes = [l.strip() for l in open('data/classes.txt')]
cls2idx = {c:i for i,c in enumerate(classes)}

def load_images(ids, folder):
    arr = np.zeros((len(ids), 32,32,3), dtype=np.uint8)
    for i,idx in enumerate(ids):
        im = Image.open(f'data/{folder}/{idx}.png').convert('RGB')
        arr[i] = np.array(im)
    return arr

train_ids = labels_df['id'].tolist()
y = labels_df['label'].map(cls2idx).values

X = load_images(train_ids, 'train_images')
print("train loaded", X.shape, time.time()-t0)

np.save('X_train_raw.npy', X)
np.save('y_train.npy', y)

test_ids = sorted([f.replace('.png','') for f in os.listdir('data/test_images')])
Xtest = load_images(test_ids, 'test_images')
print("test loaded", Xtest.shape, time.time()-t0)
np.save('X_test_raw.npy', Xtest)
with open('test_ids.txt','w') as f:
    f.write('\n'.join(test_ids))

print("classes:", classes)
print("done", time.time()-t0)
