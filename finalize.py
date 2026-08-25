import numpy as np, pandas as pd, json, joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

classes = [l.strip() for l in open('data/classes.txt')]
results = json.load(open('results.json'))
best_estimators = joblib.load('best_estimators.pkl')

# pick best model by val accuracy
best_name = max(results, key=lambda k: results[k]['val_accuracy'])
best_model = best_estimators[best_name]
print("Best model:", best_name, results[best_name])

# Predict on test set
Xtest_p = np.load('Xtest_p.npy')
test_ids = open('test_ids.txt').read().split('\n')
test_ids = [t for t in test_ids if t]

test_pred = best_model.predict(Xtest_p)
test_pred_labels = [classes[i] for i in test_pred]

sub = pd.DataFrame({'id': test_ids, 'label': test_pred_labels})
sub.to_csv('submission.csv', index=False)
print("submission saved:", sub.shape)

# Confusion matrix on validation set for best model
X_val = np.load('X_val_p.npy'); y_val = np.load('y_val.npy')
val_pred = best_model.predict(X_val)
cm = confusion_matrix(y_val, val_pred)
report = classification_report(y_val, val_pred, target_names=classes)
print(report)

with open('classification_report.txt','w') as f:
    f.write(f"Best model: {best_name}\n\n")
    f.write(report)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8,7))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix - {best_name} (val_acc={results[best_name]["val_accuracy"]:.3f})')
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j,i,cm[i,j], ha='center', va='center', fontsize=8,
                 color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("confusion matrix saved")

# Bar chart comparing all models
names = list(results.keys())
val_accs = [results[n]['val_accuracy'] for n in names]
cv_accs = [results[n]['cv_accuracy'] for n in names]
order = np.argsort(val_accs)[::-1]
names = [names[i] for i in order]
val_accs = [val_accs[i] for i in order]
cv_accs = [cv_accs[i] for i in order]

fig, ax = plt.subplots(figsize=(9,5.5))
x = np.arange(len(names)); w=0.35
ax.bar(x-w/2, cv_accs, w, label='CV Accuracy (tuning)', color='#94a3b8')
ax.bar(x+w/2, val_accs, w, label='Validation Accuracy', color='#2563eb')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')
ax.set_ylabel('Accuracy')
ax.set_title('Classifier Comparison: CIFAR-10-style Image Classification')
ax.legend()
for i,v in enumerate(val_accs):
    ax.text(i+w/2, v+0.005, f"{v:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
print("comparison chart saved")
