from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import mlxtend
from operator import itemgetter
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


from lib.logger import count_combinations, set_logger, reset_logger

SFS_ICON = '🟥'
SBS_ICON = '🟦'
SFFS_ICON = '🟨'
SBFS_ICON = '🟩'


class ClassifierHelper(object):
    def __init__(self, X_train, X_test, y_train, y_test, feature_names, sfa_enabled=True, bot_enabled=True, bot_token_key=None, bot_chat_id_key=None) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.bot_token_key = bot_token_key
        self.bot_chat_id_key = bot_chat_id_key
        self.sfa_enabled = sfa_enabled
        self.bot_enabled = bot_enabled

    def exec_random_forest(self):
        K_FOLD = 10
        feature_selection_key = 'sfs'
        param_grid = [
            {
                f'{feature_selection_key}__estimator__n_estimators': [50, 100, 150],
                f'{feature_selection_key}__estimator__criterion': ['gini', 'entropy'],
            }
        ]
        icon = '🟣'

        if self.sfa_enabled:
            self.exec_train_test(
                f'{icon} Random Forest - SFS {SFS_ICON}',
                param_grid,
                K_FOLD,
                RandomForestClassifier(),
                feature_selection_key,
                True,
                False,
            )

            self.exec_train_test(
                f'{icon} Random Forest - SBS {SBS_ICON}',
                param_grid,
                K_FOLD,
                RandomForestClassifier(),
                feature_selection_key,
                False,
                False,
            )

            self.exec_train_test(
                f'{icon} Random Forest - SFFS {SFFS_ICON}',
                param_grid,
                K_FOLD,
                RandomForestClassifier(),
                feature_selection_key,
                True,
                True,
            )

            self.exec_train_test(
                f'{icon} Random Forest - SBFS {SBFS_ICON}',
                param_grid,
                K_FOLD,
                RandomForestClassifier(),
                feature_selection_key,
                False,
                True,
            )
        else:
            self.exec_train_test(
                f'{icon} Random Forest',
                param_grid,
                K_FOLD,
                RandomForestClassifier()
            )
            

    def exec_knn(self):
        K_FOLD = 10
        clf_name = 'KNN'
        feature_selection_key = 'sfs'
        param_grid = [
            {
                f'{feature_selection_key}__estimator__n_neighbors': [5, 10, 15],
                f'{feature_selection_key}__estimator__p': [1, 2],
                f'{feature_selection_key}__estimator__weights': ['uniform', 'distance'],
            }
        ]
        icon = '🔴'

        self.exec_train_test(
            f'{icon} {clf_name} - SFS {SFS_ICON}',
            param_grid,
            K_FOLD,
            KNeighborsClassifier(n_jobs=-1),
            feature_selection_key,
            True,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBS {SBS_ICON}',
            param_grid,
            K_FOLD,
            KNeighborsClassifier(n_jobs=-1),
            feature_selection_key,
            False,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SFFS {SFFS_ICON}',
            param_grid,
            K_FOLD,
            KNeighborsClassifier(n_jobs=-1),
            feature_selection_key,
            True,
            True,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBFS {SBFS_ICON}',
            param_grid,
            K_FOLD,
            KNeighborsClassifier(n_jobs=-1),
            feature_selection_key,
            False,
            True,
            -1
        )

    def exec_decision_tree(self):
        K_FOLD = 10
        clf_name = 'Decision Tree'
        feature_selection_key = 'sfs'
        param_grid = [
            {
                f'{feature_selection_key}__estimator__criterion': ['gini', 'entropy'],
                f'{feature_selection_key}__estimator__splitter': ['best', 'random'],
            }
        ]
        icon = '🟠'

        self.exec_train_test(
            f'{icon} {clf_name} - SFS {SFS_ICON}',
            param_grid,
            K_FOLD,
            DecisionTreeClassifier(),
            feature_selection_key,
            True,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBS {SBS_ICON}',
            param_grid,
            K_FOLD,
            DecisionTreeClassifier(),
            feature_selection_key,
            False,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SFFS {SFFS_ICON}',
            param_grid,
            K_FOLD,
            DecisionTreeClassifier(),
            feature_selection_key,
            True,
            True,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBFS {SBFS_ICON}',
            param_grid,
            K_FOLD,
            DecisionTreeClassifier(),
            feature_selection_key,
            False,
            True,
        )

    def exec_ada_boost(self):
        K_FOLD = 10
        clf_name = 'AdaBoost'
        feature_selection_key = 'sfs'
        param_grid = [
            {
                f'{feature_selection_key}__estimator__n_estimators': [50, 100, 150],
                f'{feature_selection_key}__estimator__learning_rate': [0.5, 1],
            }
        ]
        icon = '🟡'

        self.exec_train_test(
            f'{icon} {clf_name} - SFS {SFS_ICON}',
            param_grid,
            K_FOLD,
            AdaBoostClassifier(),
            feature_selection_key,
            True,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBS {SBS_ICON}',
            param_grid,
            K_FOLD,
            AdaBoostClassifier(),
            feature_selection_key,
            False,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SFFS {SFFS_ICON}',
            param_grid,
            K_FOLD,
            AdaBoostClassifier(),
            feature_selection_key,
            True,
            True,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBFS {SBFS_ICON}',
            param_grid,
            K_FOLD,
            AdaBoostClassifier(),
            feature_selection_key,
            False,
            True,
        )

    def exec_gradient_boosting(self):
        K_FOLD = 10
        clf_name = 'Gradient Boosting'
        feature_selection_key = 'sfs'
        param_grid = [
            {
                f'{feature_selection_key}__estimator__n_estimators': [50, 100, 150],
                f'{feature_selection_key}__estimator__learning_rate': [0.5, 1],
                f'{feature_selection_key}__estimator__loss': ['deviance', 'exponential']
            }
        ]
        icon = '🟢'

        self.exec_train_test(
            f'{icon} {clf_name} - SFS {SFS_ICON}',
            param_grid,
            K_FOLD,
            GradientBoostingClassifier(),
            feature_selection_key,
            True,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBS {SBS_ICON}',
            param_grid,
            K_FOLD,
            GradientBoostingClassifier(),
            feature_selection_key,
            False,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SFFS {SFFS_ICON}',
            param_grid,
            K_FOLD,
            GradientBoostingClassifier(),
            feature_selection_key,
            True,
            True,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBFS {SBFS_ICON}',
            param_grid,
            K_FOLD,
            GradientBoostingClassifier(),
            feature_selection_key,
            False,
            True,
        )

    def exec_svm(self):
        K_FOLD = 10
        clf_name = 'SVM'
        feature_selection_key = 'sfs'
        param_grid = [
            {
                f'{feature_selection_key}__estimator__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                f'{feature_selection_key}__estimator__gamma': ['scale', 'auto'],
            }
        ]
        icon = '🔵'

        self.exec_train_test(
            f'{icon} {clf_name} - SFS {SFS_ICON}',
            param_grid,
            K_FOLD,
            SVC(),
            feature_selection_key,
            True,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBS {SBS_ICON}',
            param_grid,
            K_FOLD,
            SVC(),
            feature_selection_key,
            False,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SFFS {SFFS_ICON}',
            param_grid,
            K_FOLD,
            SVC(),
            feature_selection_key,
            True,
            True,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBFS {SBFS_ICON}',
            param_grid,
            K_FOLD,
            SVC(),
            feature_selection_key,
            False,
            True,
        )

    def exec_mlp(self):
        K_FOLD = 10
        clf_name = 'MLP'
        feature_selection_key = 'sfs'
        param_grid = [
            {
                f'{feature_selection_key}__estimator__hidden_layer_sizes': [100, 150],
                f'{feature_selection_key}__estimator__activation': ['relu', 'identity', 'logistic'],
                f'{feature_selection_key}__estimator__learning_rate': ['constant', 'invscaling', 'adaptive'],
            }
        ]
        icon = '⚫️'

        self.exec_train_test(
            f'{icon} {clf_name} - SFS {SFS_ICON}',
            param_grid,
            K_FOLD,
            MLPClassifier(),
            feature_selection_key,
            True,
            False,
            -1
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBS {SBS_ICON}',
            param_grid,
            K_FOLD,
            MLPClassifier(),
            feature_selection_key,
            False,
            False,
            -1
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SFFS {SFFS_ICON}',
            param_grid,
            K_FOLD,
            MLPClassifier(),
            feature_selection_key,
            True,
            True,
            -1
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBFS {SBFS_ICON}',
            param_grid,
            K_FOLD,
            MLPClassifier(),
            feature_selection_key,
            False,
            True,
            -1
        )

    def exec_logistic_regression(self):
        K_FOLD = 10
        clf_name = 'Logistic Regression'
        feature_selection_key = 'sfs'
        param_grid = [
            {
                f'{feature_selection_key}__estimator__penalty': ['none', 'l1', 'l2', 'elasticnet'],
                f'{feature_selection_key}__estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                f'{feature_selection_key}__estimator__multi_class': ['auto', 'ovr', 'multinomial'],
            }
        ]
        icon = '⚪️'

        self.exec_train_test(
            f'{icon} {clf_name} - SFS {SFS_ICON}',
            param_grid,
            K_FOLD,
            LogisticRegression(),
            feature_selection_key,
            True,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBS {SBS_ICON}',
            param_grid,
            K_FOLD,
            LogisticRegression(),
            feature_selection_key,
            False,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SFFS {SFFS_ICON}',
            param_grid,
            K_FOLD,
            LogisticRegression(),
            feature_selection_key,
            True,
            True,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBFS {SBFS_ICON}',
            param_grid,
            K_FOLD,
            LogisticRegression(),
            feature_selection_key,
            False,
            True,
        )

    def exec_xgboost(self):
        K_FOLD = 10
        clf_name = 'XGBoost'
        feature_selection_key = 'sfs'
        param_grid = [
            {
                f'{feature_selection_key}__estimator__learning_rate': [0.3, 0.5],
                f'{feature_selection_key}__estimator__max_depth': [5, 10],
            }
        ]
        icon = '🟤'

        self.exec_train_test(
            f'{icon} {clf_name} - SFS {SFS_ICON}',
            param_grid,
            K_FOLD,
            xgb.XGBClassifier(),
            feature_selection_key,
            True,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBS {SBS_ICON}',
            param_grid,
            K_FOLD,
            xgb.XGBClassifier(),
            feature_selection_key,
            False,
            False,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SFFS {SFFS_ICON}',
            param_grid,
            K_FOLD,
            xgb.XGBClassifier(),
            feature_selection_key,
            True,
            True,
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBFS {SBFS_ICON}',
            param_grid,
            K_FOLD,
            xgb.XGBClassifier(),
            feature_selection_key,
            False,
            True,
        )

    def exec_xgboost_gpu(self):
        K_FOLD = 10
        clf_name = 'XGBoost'
        feature_selection_key = 'sfs'
        param_grid = [
            {
                f'{feature_selection_key}__estimator__learning_rate': [0.3, 0.5],
                f'{feature_selection_key}__estimator__max_depth': [5, 10],
                f'{feature_selection_key}__estimator__tree_method': ['gpu_hist'],
                f'{feature_selection_key}__estimator__single_precision_histogram': [True],
            }
        ]
        icon = '🟤'

        self.exec_train_test(
            f'{icon} {clf_name} - SFS {SFS_ICON}',
            param_grid,
            K_FOLD,
            xgb.XGBClassifier(),
            feature_selection_key,
            True,
            False,
            6
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBS {SBS_ICON}',
            param_grid,
            K_FOLD,
            xgb.XGBClassifier(),
            feature_selection_key,
            False,
            False,
            6
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SFFS {SFFS_ICON}',
            param_grid,
            K_FOLD,
            xgb.XGBClassifier(),
            feature_selection_key,
            True,
            True,
            6
        )

        self.exec_train_test(
            f'{icon} {clf_name} - SBFS {SBFS_ICON}',
            param_grid,
            K_FOLD,
            xgb.XGBClassifier(),
            feature_selection_key,
            False,
            True,
            6
        )

    def exec_train_test(
            self,
            exp_id,
            param_grid,
            k_fold,
            clf_instance,
            feature_selection_key=None,
            feature_selection_forward=None,
            feature_selection_floating=None,
            n_jobs=-1):

        TIMESTAMP = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        TOTAL_COMB = count_combinations(param_grid, k_fold)
        set_logger(TIMESTAMP, exp_id, TOTAL_COMB,
                   self.bot_token_key, self.bot_chat_id_key, self.bot_enabled)

        feature_selection = SFS(
            estimator=clf_instance,
            k_features=(1, self.X_train.shape[1]),
            scoring='accuracy',
            cv=k_fold,
            n_jobs=n_jobs,
            forward=feature_selection_forward,
            floating=feature_selection_floating,
            verbose=2
        )

        pipeline_steps = [(feature_selection_key, feature_selection), ('clf', clf_instance)] if self.sfa_enabled else [('clf', clf_instance)]
        pipe_instance = Pipeline(pipeline_steps)

        grid_search = GridSearchCV(
            estimator=pipe_instance,
            param_grid=param_grid,
            scoring='accuracy',
            cv=k_fold,
            refit=True,
            verbose=2,
            error_score='raise'
        )

        try:
            print(f'Starting GridSearchCV...')
            grid_search.fit(self.X_train, self.y_train)
            print('Done successfully!')

            if self.sfa_enabled:
                print(
                    f'Best features: {itemgetter(*grid_search.best_estimator_.steps[0][1].k_feature_idx_)(self.feature_names)}')

            print(f'Best params: {grid_search.best_params_}')
            print(f'Acc: {grid_search.best_score_}')

            clf_pred = grid_search.best_estimator_[1].predict(
                self.X_test[:, list(grid_search.best_estimator_.steps[0][1].k_feature_idx_)])
            print(
                f'Metrics: {self.calculate_classification_metrics(clf_pred)}')
        except Exception as e:
            print(f'Algo deu errado: {e}')
        finally:
            reset_logger()

    def calculate_classification_metrics(self, y_pred):
        """
        Function that calculates the classification results.

        Parameters
        ----------
        y_test : np.array
            Test target.
        y_pred : np.array
            Predicted target.

        Returns
        -------
        dict
            Dictionary with the results of the classification.
        """

        # Calculate metrics
        clf_report = classification_report(
            self.y_test, y_pred, output_dict=True)
        true_class_results, false_class_results, accuracy, macro_avg, weighted_avg = itemgetter(
            '0.0', '1.0', 'accuracy', 'macro avg', 'weighted avg')(clf_report)
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        return {
            'accuracy': accuracy,
            'macro_avg': macro_avg,
            'weighted_avg': weighted_avg,
            'true_class_results': true_class_results,
            'false_class_results': false_class_results,
            'conf_matrix': conf_matrix
        }
