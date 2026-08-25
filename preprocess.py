import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import time, joblib

t0=time.time()
X = np.load('X_train_raw.npy').reshape(29400, -1).astype(np.float32) / 255.0
y = np.load('y_train.npy')
Xtest = np.load('X_test_raw.npy').reshape(7600, -1).astype(np.float32) / 255.0

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
print("split", X_train.shape, X_val.shape, time.time()-t0)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
Xtest_s = scaler.transform(Xtest)
print("scaled", time.time()-t0)

pca = PCA(n_components=100, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_val_p = pca.transform(X_val_s)
Xtest_p = pca.transform(Xtest_s)
print("pca explained var:", pca.explained_variance_ratio_.sum(), time.time()-t0)

np.save('X_train_p.npy', X_train_p)
np.save('X_val_p.npy', X_val_p)
np.save('Xtest_p.npy', Xtest_p)
np.save('y_train_split.npy', y_train)
np.save('y_val.npy', y_val)
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
print("done", time.time()-t0)
