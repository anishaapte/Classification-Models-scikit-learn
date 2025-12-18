from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Classifiers():
    def __init__(self,data):
        ''' 
        TODO: Write code to convert the given pandas dataframe into training and testing data 
        # all the data should be nxd arrays where n is the number of samples and d is the dimension of the data
        # all the labels should be nx1 vectors with binary labels in each entry 
        '''
        #Loading the dataset
        data_set = pd.read_csv("input.csv")
#        print("Total rows:", df.shape[0])

        #Constructing a scatter plot - done once then commented out
#        class0 = data_set[data_set["label"] == 0]
#        class1 = data_set[data_set["label"] == 1]
#        
#        plt.figure(figsize = (8,8))
#        plt.scatter(class0["A"], class0["B"], marker = "o", color = "blue", label = "Class 0")
#        plt.scatter(class1["A"], class1["B"], marker = "x", color = "red", label = "Class 1")
#        
#        plt.title("Dataset")
#        plt.ylabel("A")
#        plt.xlabel("B")
#        plt.legend()
#        plt.grid(True)
        
#        plt.show()
        
        
        # Splitting the dataset into training (60%) and testing (40%), confirming split done after split(commented out)
        inputs = data[["A","B"]].values
        labels = data["label"].values
        
        train_input, test_input, train_output, test_output = train_test_split(
            inputs, labels, test_size=0.4, random_state=0
        )
        
        self.training_data = train_input
        self.training_labels = train_output
        self.testing_data = test_input
        self.testing_labels = test_output
        self.outputs = []
        
#        print("Training data shape:", self.training_data.shape)
#        print("Testing data shape:", self.testing_data.shape)
#        print("Training labels shape:", self.training_labels.shape)
#        print("Testing labels shape:", self.testing_labels.shape)
    
    def test_clf(self, clf, classifier_name=''):
        # TODO: Fit the classifier and extrach the best score, training score and parameters
        pass
        # Use the following line to plot the results
        # self.plot(self.testing_data, clf.predict(self.testing_data),model=clf,classifier_name=name)

    def classifyNearestNeighbors(self):
        # TODO: Write code to run a Nearest Neighbors classifier
#        # defining the params like given in document
        define_param = {
            "n_neighbors": list(range(1, 20, 2)),
            "leaf_size" : list(range(5,35,5))
        }
        
        #Initialize model
        knn = KNeighborsClassifier()
        
        #Setting up cross validation
        grid = GridSearchCV (
            knn,
            define_param,
            cv = 5,
            scoring = "accuracy"
        )
        
        #Fit model on training
        grid.fit(self.training_data,self.training_labels)
        
        # Get best model from GridSearch
        knn_best = grid.best_estimator_
        
        # Find best training score, which is cross-validated
        best_score_training = grid.best_score_
        
        # Test the best model on the held-out test data
        test_score = knn_best.score(self.testing_data, self.testing_labels)
        
        # Saving to output.csv and plotting
        result = f"KNN, {best_score_training:.4f}, {test_score:.4f}"
        self.outputs.append(result)
        
        self.plot(self.testing_data, self.testing_labels, knn_best, classifier_name="KNN")
        
        
        #Test
#        print(f"Best Parameters: {grid.best_params_}")
#        print(f"Best Training Accuracy: {best_score_training:.4f}")
#        print(f"Testing Accuracy: {test_score:.4f}")
        
        
    def classifyLogisticRegression(self):
        # TODO: Write code to run a Logistic Regression classifier
        
        # defining the params like given in document
        define_param = {
            "C": [0.1, 0.5, 1, 5, 10, 50, 100],
        }
        
        #Initialize
        log_regression = LogisticRegression(max_iter = 1000)
        
        #Setting up cross validation
        grid = GridSearchCV (
            log_regression,
            define_param,
            cv = 5,
            scoring = "accuracy"
        )
        
        #Fit model on training
        grid.fit(self.training_data,self.training_labels)
        
        # Get best model from GridSearch
        regression_best = grid.best_estimator_
        
        # Find best training score, which is cross-validated
        best_score_training = grid.best_score_
        
        # Test the best model on the held-out test data
        test_score = regression_best.score(self.testing_data,self.testing_labels)
        
        # Saving to output.csv and plotting
        result = f"Logistic Regression, {best_score_training:.4f}, {test_score:.4f}"
        self.outputs.append(result)
        
        self.plot(self.testing_data, self.testing_labels, regression_best, classifier_name="Logistic Regression")
        
        
        # Test
#        print(f"Best Parameters: {grid.best_params_}")
#        print(f"Best Training Accuracy: {best_score_training:.4f}")
#        print(f"Testing Accuracy: {test_score:.4f}")
        
        
    
    def classifyDecisionTree(self):
        # TODO: Write code to run a Logistic Regression classifier
        
        # defining the params like given in document
        define_param = {
            "max_depth" : list(range(1,51)),
            "min_samples_split" : list(range(2,11))
        }
        #Initialize
        tree = DecisionTreeClassifier(random_state = 0)
        
        #Setting up cross validation
        grid = GridSearchCV (
            tree,
            define_param,
            cv = 5,
            scoring = "accuracy"
        )
        
        #Fit model on training
        grid.fit(self.training_data,self.training_labels)
        
        # Get best model from GridSearch
        tree_best = grid.best_estimator_
        
        # Find best training score, which is cross-validated
        best_score_training = grid.best_score_
        
        # Test the best model on the held-out test data
        test_score = tree_best.score(self.testing_data,self.testing_labels)
        
        # Saving to output.csv and plotting
        result = f"Decision Tree, {best_score_training:.4f}, {test_score:.4f}"
        self.outputs.append(result)
        
        self.plot(self.testing_data, self.testing_labels, tree_best, classifier_name="Decision Tree")
        
        
        # Test
#        print(f"Best Parameters: {grid.best_params_}")
#        print(f"Best Training Accuracy: {best_score_training:.4f}")
#        print(f"Testing Accuracy: {test_score:.4f}")


    def classifyRandomForest(self):
        # TODO: Write code to run a Random Forest classifier
        
        # defining the params like given in document
        define_param = {
            "max_depth" : list(range(1,6)),
            "min_samples_split" : list(range(2,11))
        }
        #Initialize
        forest = RandomForestClassifier(random_state = 0)
        
        #Setting up cross validation
        grid = GridSearchCV (
            forest,
            define_param,
            cv = 5,
            scoring = "accuracy"
        )
        
        #Fit model on training
        grid.fit(self.training_data,self.training_labels)
        
        # Get best model from GridSearch
        forest_best = grid.best_estimator_
        
        # Find best training score, which is cross-validated
        best_score_training = grid.best_score_
        
        # Test the best model on the held-out test data
        test_score = forest_best.score(self.testing_data,self.testing_labels)
        
        # Saving to output.csv and plotting
        result = f"Random Forest, {best_score_training:.4f}, {test_score:.4f}"
        self.outputs.append(result)
        
        self.plot(self.testing_data, self.testing_labels, forest_best, classifier_name="Random Forest")
        
        
        # Test
#        print(f"Best Parameters: {grid.best_params_}")
#        print(f"Best Training Accuracy: {best_score_training:.4f}")
#        print(f"Testing Accuracy: {test_score:.4f}")


    def classifyAdaBoost(self):
        # TODO: Write code to run a AdaBoost classifier
        
        # defining the params like given in document
        define_param = {
            "n_estimators" : list(range(10,80,10))
        }
        #Initialize
        boost = AdaBoostClassifier(random_state = 0)
        
        #Setting up cross validation
        grid = GridSearchCV (
            boost,
            define_param,
            cv = 5,
            scoring = "accuracy"
        )
        
        #Fit model on training
        grid.fit(self.training_data,self.training_labels)
        
        # Get best model from GridSearch
        boost_best = grid.best_estimator_
        
        # Find best training score, which is cross-validated
        best_score_training = grid.best_score_
        
        # Test the best model on the held-out test data
        test_score = boost_best.score(self.testing_data,self.testing_labels)
        
        # Saving to output.csv and plotting
        result = f"AdaBoost, {best_score_training:.4f}, {test_score:.4f}"
        self.outputs.append(result)
        
        self.plot(self.testing_data, self.testing_labels, boost_best, classifier_name="AdaBoost")
        
        
        # Test
#        print(f"Best Parameters: {grid.best_params_}")
#        print(f"Best Training Accuracy: {best_score_training:.4f}")
#        print(f"Testing Accuracy: {test_score:.4f}")

        

    def plot(self, X, Y, model,classifier_name = ''):
        X1 = X[:, 0]
        X2 = X[:, 1]

        X1_min, X1_max = min(X1) - 0.5, max(X1) + 0.5
        X2_min, X2_max = min(X2) - 0.5, max(X2) + 0.5

        X1_inc = (X1_max - X1_min) / 200.
        X2_inc = (X2_max - X2_min) / 200.

        X1_surf = np.arange(X1_min, X1_max, X1_inc)
        X2_surf = np.arange(X2_min, X2_max, X2_inc)
        X1_surf, X2_surf = np.meshgrid(X1_surf, X2_surf)

        L_surf = model.predict(np.c_[X1_surf.ravel(), X2_surf.ravel()])
        L_surf = L_surf.reshape(X1_surf.shape)

        plt.title(classifier_name)
        plt.contourf(X1_surf, X2_surf, L_surf, cmap = plt.cm.coolwarm, zorder = 1)
        plt.scatter(X1, X2, s = 38, c = Y)

        plt.margins(0.0)
        # uncomment the following line to save images
        plt.savefig(f'{classifier_name}.png')
        plt.show()

    
if __name__ == "__main__":
    df = pd.read_csv('input.csv')
    models = Classifiers(df)
    print('Classifying with NN...')
    models.classifyNearestNeighbors()
    print('Classifying with Logistic Regression...')
    models.classifyLogisticRegression()
    print('Classifying with Decision Tree...')
    models.classifyDecisionTree()
    print('Classifying with Random Forest...')
    models.classifyRandomForest()
    print('Classifying with AdaBoost...')
    models.classifyAdaBoost()

    with open("output.csv", "w") as f:
        print('Name, Best Training Score, Testing Score',file=f)
        for line in models.outputs:
            print(line, file=f)
