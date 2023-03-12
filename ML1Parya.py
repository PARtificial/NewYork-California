# ML Assignment1
# By Parya Payami
# This code uses scikit-learn library to evaluate decision trees.
# The datasets are ID = 823 and ID = 44090 from openml website

# Importing functions
from sklearn import tree, model_selection, datasets, metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

def subtask1(data, param_range):
    x, y = data.data, data.target

    # Initialize lists to store the mean roc_auc scores 
    train_scores = []
    test_scores = []

    for param in param_range:
        # Create a decision tree classifier with the range of values of min_samples_leaf
        model = tree.DecisionTreeClassifier(min_samples_leaf=param, criterion="entropy")
        # Calculate the cross-validated roc_auc scores on the entire dataset
        cv = model_selection.cross_validate(model, x, y, scoring="roc_auc", cv=10, return_train_score=True)
        # Append the mean roc_auc score for test and train to their lists
        train_scores.append(np.mean(cv["train_score"]))
        test_scores.append(np.mean(cv["test_score"]))
        
    # Plot graph of min_samples_leaf and mean roc_auc score of the 10 folds
    plt.plot(param_range, train_scores, label='Training scores')
    plt.plot(param_range, test_scores, label='Test scores')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Mean ROC AUC Score')
    plt.title("Subtask 1")
    plt.legend()
    plt.show()
    
def subtask2(data, param_range):
    x, y = data.data, data.target
    param_grid = {'min_samples_leaf': param_range}
    model = tree.DecisionTreeClassifier()
    tuned_model= model_selection.GridSearchCV(model, param_grid, scoring="roc_auc", cv=10)
    tuned_model.fit(x, y)
    plt.axvline(x = tuned_model.best_params_['min_samples_leaf'], color = 'grey', linestyle='dashed', label = 'Optimal leaf')
    best_param = tuned_model.best_params_['min_samples_leaf']
    mean_scores = tuned_model.cv_results_['mean_test_score']
    
    # Plot mean scores in a graph
    plt.plot(param_range, tuned_model.cv_results_['mean_test_score'])
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Mean ROC AUC Score')
    plt.title("Subtask 2")
    plt.legend()
    plt.show()

    # Print mean scores in a table
    print("The mean roc_auc score of the 10-folds:")
    for mean, params in zip(mean_scores, tuned_model.cv_results_['params']):
        print("%0.3f for %r" % (mean, params))
    print("The best parameter value for dataset1 is", best_param)
    print(" ")
    
# Assigining min number of leaf
param_range_1 = [250, 300, 350, 400, 450]
param_range_2 = [10, 20, 30, 40, 50]

# Importing datasets
house = datasets.fetch_openml(data_id=823, parser='auto')
california = datasets.fetch_openml(data_id=44090, parser='auto')

if __name__ == '__main__':
    subtask1(house, param_range_1)
    subtask2(house, param_range_1)
    subtask1(california, param_range_2)
    subtask2(california, param_range_2)
