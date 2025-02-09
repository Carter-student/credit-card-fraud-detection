from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb
import numpy as np
from data_preprocessing import DataPreprocessor
from sklearn.metrics import f1_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


data_preprocessor = DataPreprocessor(columns_to_normalise=[])
X_train, X_test, y_train, y_test = data_preprocessor.run_preprocess()

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2]
}

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=y_train[y_train==0].shape[0]/ y_train[y_train == 1].shape[0] 
    )


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_grid,
    n_iter=20, 
    scoring="average_precision",
    cv=cv,
    verbose=1,
    n_jobs=-1,
    refit=True
)

search = random_search.fit(X_train, y_train)

# Best parameters
print("Best Hyperparameters:", random_search.best_params_)


y_pred = search.best_estimator_.predict(X_test)

f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1:.4f}")


y_pred_proba = [prob[1] for prob in search.best_estimator_.predict_proba(X_test)]

# Compute AUC-PR (Precision-Recall AUC)
auc_pr = average_precision_score(y_test, y_pred_proba)
print(f"AUC-PR (Average Precision): {auc_pr:.4f}")

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f"AUC-PR: {auc_pr:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()
