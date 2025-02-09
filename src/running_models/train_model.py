from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb
import numpy as np
from data_preprocessing import DataPreprocessor
from sklearn.metrics import f1_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from constants import RANDOM_STATE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from typing import Optional


class XgboostModel:
    def __init__(
        self,
        columns_to_normalise: Optional[list] = None,
        use_smote: bool = False
    ):
        if columns_to_normalise is None:
            columns_to_normalise = []

        self.use_smote = use_smote
        data_preprocessor = DataPreprocessor(columns_to_normalise=columns_to_normalise)
        self.X_train, self.X_test, self.y_train, self.y_test = data_preprocessor.run_preprocess()
        


    def cross_validation(
        self
    ):
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "gamma": [0, 0.1, 0.2]
        }
        
        weighting_kwarg = {}
        if not self.use_smote:
            weighting_kwarg = {
                'scale_pos_weight':self.y_train[self.y_train==0].shape[0]/ self.y_train[self.y_train == 1].shape[0]
            }
            param_grid.update({"subsample": [0.7, 0.8, 1.0]})
            
        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            **weighting_kwarg
        )
        
        if self.use_smote:
            pipeline_smote = Pipeline([
            ("smote", SMOTE(sampling_strategy="auto", random_state=RANDOM_STATE)),
            ("xgb", xgb_clf)
            ])
            param_grid = {'xgb__' + key: value for key, value in param_grid.items()}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        random_search = RandomizedSearchCV(
            estimator=pipeline_smote if self.use_smote else xgb_clf,
            param_distributions=param_grid,
            n_iter=20, 
            scoring="average_precision",
            cv=cv,
            verbose=1,
            n_jobs=-1,
            refit=True,
            random_state=RANDOM_STATE
        )

        search = random_search.fit(self.X_train, self.y_train)
        return search


    def reporting(
        self,
        search
    ):
        # Best parameters
        print("Best Hyperparameters:", search.best_params_)


        y_pred = search.best_estimator_.predict(self.X_test)

        f1 = f1_score(self.y_test, y_pred)
        print(f"F1-score: {f1:.4f}")

        weighted_f1 = f1_score(self.y_test, y_pred, average="weighted")
        print(f"Weighted F1-score: {weighted_f1:.4f}")

        y_pred_proba = [prob[1] for prob in search.best_estimator_.predict_proba(self.X_test)]

        # Compute AUC-PR (Precision-Recall AUC)
        auc_pr = average_precision_score(self.y_test, y_pred_proba)
        print(f"AUC-PR (Average Precision): {auc_pr:.4f}")

        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label=f"AUC-PR: {auc_pr:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    xgbm_smote = XgboostModel(use_smote=True)
    search_result_smote = xgbm_smote.cross_validation()
    best_score_smote = search_result_smote.best_score_
    print(f'Best Score cross validation smote {best_score_smote:.4f}')
    
    xgbm = XgboostModel()
    search_result = xgbm.cross_validation()
    best_score_scaling = search_result.best_score_
    print(f'Best Score cross validation scale_pos_weight {best_score_scaling:.4f}')
    

    
    if best_score_scaling > best_score_smote:
        xgbm.reporting(search_result)
        print('then smote just for show')
        xgbm_smote.reporting(search_result_smote)
    else:
        xgbm_smote.reporting(search_result_smote)
        print('then scaling just for show')
        xgbm.reporting(search_result)


