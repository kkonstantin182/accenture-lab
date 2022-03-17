import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import os



SEED = 42

class HyperTuner():
    
    def __init__(self, X, y, model):

        self.X = X
        self.y = y
        #self.X_val = X_val
        #self.y_val = y_val
        self.model = model
        self.classifier = None
        self.params = None
        self.results = None
        self.best_params_and_results = None
    
    @staticmethod
    def get_available_models():

        return ('RandomForestClassifier')

    def _init_classifier_params(self):

        if self.model not in self.get_available_models():
            raise Exception(
                "This model is not available.\n" 
                "Use 'get_available_models()' to get the list of the available models.")

        elif self.model == 'RandomForestClassifier':
            #print('hui')

            self.classifier = RandomForestClassifier(random_state=SEED)
            
            n_estimators = [10, 50, 100, 200, 500]
            criterion = ['gini', 'entropy']
            max_depth = [int(x) for x in range(10, 120, 10)]
            max_depth.append(None)
            min_samples_split = [2, 4, 6]
            min_samples_leaf = [1, 2, 3]
            max_features = ['auto', 'sqrt', 'log2'],
            bootstrap = [True, False]

            self.params = {
                'n_estimators' : n_estimators, # work
                'criterion' : criterion, # work
                'max_depth' : max_depth, # work
                'min_samples_split' : min_samples_split, # work
                'min_samples_leaf' : min_samples_leaf, # work
                #'max_features' : max_features, # don't work
                'bootstrap' : bootstrap
            }

    def fit_predict(self):

        self._init_classifier_params()
        self.results = {}
        self.best_params_and_results = []

        gs = GridSearchCV(self.classifier, self.params, cv=5, return_train_score=True)
        gs.fit(self.X, self.y)

        self.results = gs.cv_results_
        
        self.best_params_and_results.append(gs.cv_results_['params'][gs.best_index_]) # the best set of parameters
        self.best_params_and_results.append(gs.best_score_) # the best test score
        self.best_params_and_results.append(gs.cv_results_['mean_train_score'][gs.best_index_]) # the best train score

        self._save_results()

        self._get_best_estimaror()

    def _save_results(self):

        abs_path = os.getcwd()
        directory = os.path.join(abs_path, 'hyp_tun_res')
        date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        if not os.path.exists(directory):
             os.makedirs(directory)
        path1 = os.path.join(directory, f'results_{date}.npy') 
        path2 = os.path.join(directory, f'best_results_{date}.txt') 
        np.save(path1, self.results)
        np.savetxt(path2, self.best_params_and_results, fmt='%5s', delimiter=',')
        
    def _get_best_estimaror(self):

        print("The best set of parameteres is: " + str(self.best_params_and_results[0]))
        print("The best test score is : " + str(self.best_params_and_results[1]))
        print("The best train score is : " + str(self.best_params_and_results[2]))
        