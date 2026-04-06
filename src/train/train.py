
# CLASSICAL ML WITH ADA AND BERT

import pandas as pd
import pickle

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn import svm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from src.utils.functions import emb_set_builder


def trainer_evaluator(df_train: pd.DataFrame, df_test: pd.DataFrame, embedding_col: str, tag_col: str) -> None:
    """
    Trains and evaluates multiple classification models using grid search.

    Args:
        df_train: Training data DataFrame.
        df_test: Test data DataFrame.
        embedding_col: Name of the column containing the embeddings.
        tag_col: Name of the column containing the target labels.

    Returns:
        None
    """

    print(f'\n TRAINING {embedding_col} \n')
    train = emb_set_builder(df_train, embedding_col, tag_col).dropna()
    test = emb_set_builder(df_test, embedding_col, tag_col).dropna()

    # Test-train split
    X_train = train.drop('tag', axis=1)
    y_train = train[['tag']]
    X_test = test.drop('tag', axis=1)
    y_test = test[['tag']]


    print(X_train.shape, X_test.shape)
    print(f'\nLabel distribution train:\nLabel 1:{y_train[y_train.tag == 1].shape[0]}\nLabel 0:{y_train[y_train.tag == 0].shape[0]}')
    print(f'\nLabel distribution test:\nLabel 1:{y_test[y_test.tag == 1].shape[0]}\nLabel 0:{y_test[y_test.tag == 0].shape[0]}')

    # Label encoding
    lb = preprocessing.LabelEncoder()
    lb.fit(y_train.tag.ravel())
    y_train = lb.transform(y_train.tag.ravel())
    y_test = lb.transform(y_test.tag.ravel())

    # Setting model pipelines
    rand_st = 111
    pipe_lr = Pipeline([('clf', LogisticRegression(random_state=rand_st))])
    pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state=rand_st))])
    pipe_svm = Pipeline([('clf', svm.SVC(random_state=rand_st, probability=True))])
    # pipe_gb = Pipeline([('clf', GradientBoostingClassifier(random_state=rand_st))])
    pipe_xgb = Pipeline([('clf', XGBClassifier(random_state=rand_st))])

    # Setting grid search parameters
    param_range_depth = [1, 2, 3, 5]
    param_trees = [100, 200, 500]
    param_range_fl = [1.0, 0.5, 0.1]

    grid_params_lr = [{'clf__penalty': ['l1', 'l2']}]
    grid_params_rf = [{'clf__criterion': ['gini', 'entropy'], 'clf__n_estimators': param_trees,
                       'clf__max_depth': param_range_depth}]
    grid_params_svm = [{'clf__kernel': ['linear', 'rbf'], 'clf__C': param_range_fl}]
    # grid_params_gb = [{'clf__n_estimators': param_trees, 'clf__max_depth': param_range_depth}]
    grid_params_xgb = [{'clf__max_depth': param_range_depth}]

    score = 'f1_macro'
    cv = 5

    # Construct grid searches
    jobs = -1
    gs_lr = GridSearchCV(estimator=pipe_lr, param_grid=grid_params_lr, scoring=score, cv=cv)
    gs_rf = GridSearchCV(estimator=pipe_rf, param_grid=grid_params_rf, scoring=score, cv=cv, n_jobs=jobs)
    gs_svm = GridSearchCV(estimator=pipe_svm, param_grid=grid_params_svm, scoring=score, cv=cv, n_jobs=jobs)
    # gs_gb = GridSearchCV(estimator=pipe_gb, param_grid=grid_params_gb, scoring=score, cv=cv, n_jobs=jobs)
    gs_hgb = GridSearchCV(estimator=pipe_xgb, param_grid=grid_params_xgb, scoring=score, cv=cv, n_jobs=jobs)

    # List of pipelines for ease of iteration
    grids = [gs_lr, gs_rf, gs_svm, gs_hgb]

    # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {0: "Logistic Regression", 1: 'Random Forest', 2: "SVM", 3: 'XGBoost'}

    # Training and grid search
    print('Performing model optimizations...')
    best_metric = 0.0
    best_clf = 0
    best_gs = ''
    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])
        # Fit grid search
        gs.fit(X_train, y_train)
        # Best params
        print('Best params: %s' % gs.best_params_)
        # Best training data accuracy
        print('Best training accuracy: %.3f' % gs.best_score_)
        # Predict on test data with best params
        y_pred = gs.predict(X_test)
        # Test data accuracy of model with best params
        print('Test set f1 score for best params: %.3f ' % f1_score(y_test, y_pred))
        print('\n')
        print("=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))
        print('\n')
        print("=== Classification Report ===")
        print(classification_report(y_test, y_pred))

        # Track best (the highest test f1 score) model
        if f1_score(y_test, y_pred) > best_metric:
            best_metric = f1_score(y_test, y_pred)
            best_gs = gs
            best_clf = idx

    print('\n' + 50*'-')
    print('Classifier with best test set accuracy %s' % grid_dict[best_clf])
    print('\n' + 50*'-')

    # Best model
    model = best_gs.best_estimator_

    # Evaluation

    with open(f'../../models/{embedding_col}_{model}.pkl', 'wb') as file:
        pickle.dump(model, file)

    y_test = pd.DataFrame(y_test, index=X_test.index, columns=['label'])
    y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=['predicted_label'])
    y_pred_prob = pd.DataFrame(model.predict_proba(X_test), index=X_test.index,
                               columns=['probability_0', 'probability_1'])
    eval_df = y_test.join(y_pred).join(y_pred_prob)

    # Performance of base model
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Threshold optimization
    prob_list = [0.3, 0.4, 0.5, 0.6, 0.7]

    for prob in prob_list:
        df_res_extreme = eval_df.copy()
        df_res_extreme['cut_label'] = df_res_extreme['probability_1'].apply(lambda x: 1 if x >= prob else 0)
        y_test_ext = df_res_extreme.label
        y_pred_ext = df_res_extreme.cut_label

        print(f'Probability: {prob}')
        print("=== Confusion Matrix ===")
        print(confusion_matrix(y_test_ext, y_pred_ext))
        print("=== Classification Report ===")
        print(classification_report(y_test_ext, y_pred_ext, zero_division=0))


if __name__ == "__main__":
    train = pd.read_pickle('../../data/tagged_set_emb.pkl')
    test = pd.read_pickle('../../data/val_set_emb.pkl')
    trainer_evaluator(train, test, 'bert', 'tag')

# python -W ignore train.py > ada_report.txt