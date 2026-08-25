import numpy as np, pandas as pd, time, json, joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

X_train = np.load('X_train_p.npy'); y_train = np.load('y_train_split.npy')
X_val = np.load('X_val_p.npy'); y_val = np.load('y_val.npy')
classes = [l.strip() for l in open('data/classes.txt')]

results = {}
best_estimators = {}
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def tune_and_eval(name, estimator, param_grid, tune_n, final_n, cv=cv3):
    t0 = time.time()
    idx = np.random.RandomState(42).choice(len(X_train), size=min(tune_n, len(X_train)), replace=False)
    gs = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy', n_jobs=1, verbose=0)
    gs.fit(X_train[idx], y_train[idx])
    tune_time = time.time() - t0
    print(f"[{name}] tuned in {tune_time:.1f}s on {len(idx)} samples -> best_params={gs.best_params_}, cv_acc={gs.best_score_:.4f}")

    # Final fit on larger subset with best params
    t1 = time.time()
    fidx = np.random.RandomState(1).choice(len(X_train), size=min(final_n, len(X_train)), replace=False)
    final_model = gs.best_estimator_.__class__(**gs.best_params_) if not isinstance(gs.best_estimator_, xgb.XGBClassifier) else xgb.XGBClassifier(**gs.best_params_, n_jobs=1, tree_method='hist', eval_metric='mlogloss')
    final_model.fit(X_train[fidx], y_train[fidx])
    fit_time = time.time() - t1

    val_pred = final_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"[{name}] final fit on {len(fidx)} samples in {fit_time:.1f}s -> val_acc={val_acc:.4f}")

    results[name] = {
        'best_params': gs.best_params_,
        'cv_accuracy': gs.best_score_,
        'val_accuracy': val_acc,
        'tune_time_s': tune_time,
        'fit_time_s': fit_time,
        'tune_n': len(idx),
        'final_n': len(fidx),
    }
    best_estimators[name] = final_model
    return final_model

# 1. SVM
tune_and_eval('SVM', SVC(kernel='rbf', probability=False, random_state=42),
              {'C':[1,10,50], 'gamma':['scale',0.01]},
              tune_n=6000, final_n=24990)

# 2. Decision Tree
tune_and_eval('DecisionTree', DecisionTreeClassifier(random_state=42),
              {'max_depth':[10,20,None], 'min_samples_split':[2,10]},
              tune_n=8000, final_n=24990)

# 3. Random Forest
tune_and_eval('RandomForest', RandomForestClassifier(random_state=42, n_jobs=1),
              {'n_estimators':[100,200], 'max_depth':[None,20]},
              tune_n=8000, final_n=24990)

# 4. KNN
tune_and_eval('KNN', KNeighborsClassifier(n_jobs=1),
              {'n_neighbors':[3,5,7], 'weights':['uniform','distance']},
              tune_n=10000, final_n=24990)

# 5. Gradient Boosting (sklearn) - slow, smaller budget
tune_and_eval('GradientBoosting', GradientBoostingClassifier(random_state=42),
              {'n_estimators':[50,100], 'learning_rate':[0.1,0.2]},
              tune_n=3000, final_n=8000)

# 6. XGBoost
tune_and_eval('XGBoost', xgb.XGBClassifier(n_jobs=1, tree_method='hist', eval_metric='mlogloss', random_state=42),
              {'n_estimators':[100,200], 'max_depth':[3,6], 'learning_rate':[0.1,0.2]},
              tune_n=8000, final_n=24990)

with open('results.json','w') as f:
    json.dump(results, f, indent=2, default=str)

joblib.dump(best_estimators, 'best_estimators.pkl')
print("\n=== SUMMARY ===")
for name, r in sorted(results.items(), key=lambda x: -x[1]['val_accuracy']):
    print(f"{name}: val_acc={r['val_accuracy']:.4f}  cv_acc={r['cv_accuracy']:.4f}  params={r['best_params']}")
