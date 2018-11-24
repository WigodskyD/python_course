'''
Assignment #7
1. Add / modify code ONLY between the marked areas (i.e. "Place code below")
2. Run the associated test harness for a basic check on completeness. A successful run of the test cases does not 
    guarantee accuracy or fulfillment of the requirements. Please do not submit your work if test cases fail.
3. To run unit tests simply use the below command after filling in all of the code:
    python 07_assignment.py

4. Unless explicitly stated, please do not import any additional libraries but feel free to use built-in Python packages
5. Submissions must be a Python file and not a notebook file (i.e *.ipynb)
6. Do not use global variables unless stated to do so
7. Make sure your work is committed to your master branch in Github
Packages required:
pip install cloudpickle==0.5.6 (this is an optional install to help remove a deprecation warning message from sklearn)
pip install sklearn
'''
# core
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ml
from sklearn import datasets as ds
from sklearn import linear_model as lm
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split as tts

# infra
import unittest


# ------ Place code below here \/ \/ \/ ------
# import plotly library and enter credential info here
import plotly
# If plotly credentials are not already saved, enter them here.
import plotly.plotly as py
import plotly.graph_objs as go


# ------ Place code above here /\ /\ /\ ------


# ------ Place code below here \/ \/ \/ ------
# Load datasets here once and assign to variables iris and boston

iris = ds.load_iris()
boston = ds.load_boston()

# ------ Place code above here /\ /\ /\ ------


# 10 points
def exercise01():
    '''
        Data set: Iris
        Return the first 5 rows of the data including the feature names as column headings in a DataFrame and a
        separate Python list containing target names
    '''

    # ------ Place code below here \/ \/ \/ ------
    target_names = iris['target_names']
    df_first_five_rows = pd.DataFrame(iris['data'][0:5])
    df_first_five_rows.columns = iris.feature_names
    # ------ Place code above here /\ /\ /\ ------

    return df_first_five_rows, target_names


# 15 points
def exercise02(new_observations):
    '''
        Data set: Iris
        Fit the Iris dataset into a kNN model with neighbors=5 and predict the category of observations passed in 
        argument new_observations. Return back the target names of each prediction (and not their encoded values,
        i.e. return setosa instead of 0).
    '''

    # ------ Place code below here \/ \/ \/ ------
    knn_black_box = KNN(n_neighbors=5)
    knn_black_box.fit(iris['data'], iris['target'])
    iris_predictions = pd.Series(knn_black_box.predict(new_observations).astype('str'))
    iris_predictions = iris_predictions.str.replace('0', 'setosa')
    iris_predictions = iris_predictions.str.replace('1', 'versicolor')
    iris_predictions = iris_predictions.str.replace('2', 'virginica')
    iris_predictions = list(iris_predictions)
    # ------ Place code above here /\ /\ /\ ------

    return iris_predictions


# 15 points
def exercise03(neighbors, split):
    '''
        Data set: Iris
        Split the Iris dataset into a train / test model with the split ratio between the two established by 
        the function parameter split.
        Fit KNN with the training data with number of neighbors equal to the function parameter neighbors
        Generate and return back an accuracy score using the test data was split out
    '''
    random_state = 21

    # ------ Place code below here \/ \/ \/ ------
    def accuracy(a, b):
        matches = 0
        for i in range(len(a)):
            if a[i] == b[i]:
                matches += 1
        return matches / len(a)
    data_train, data_test, target_train, target_test = tts(iris['data'], iris['target'],
                                                           test_size=split, random_state=random_state,
                                                           stratify=iris['target'])
    knn_black_box_train = KNN(n_neighbors=neighbors)
    knn_black_box_train.fit(data_train, target_train)
    test_prediction = knn_black_box_train.predict(data_test)
    knn_score = accuracy(test_prediction, target_test)

    # ------ Place code above here /\ /\ /\ ------

    return knn_score


# 20 points
def exercise04():
    '''
        Data set: Iris
        Generate an overfitting / underfitting curve of kNN each of the testing and training accuracy performance scores series
        for a range of neighbor (k) values from 1 to 30 and plot the curves (number of neighbors is x-axis, performance score 
        is y-axis on the chart). Return back the plotly url.
    '''

    # ------ Place code below here \/ \/ \/ ------
    # This code would need to be repeated to allow exercise04 to run in the test framework
    random_state = 21

    def accuracy(a, b):
        matches = 0
        for i in range(len(a)):
            if a[i] == b[i]:
                matches += 1
        return matches / len(a)
    # -----------------------------

    def accuracy_from_set(train_values, train_targets, chosen_values, chosen_target, neighbors_num):
        knn_model_maker = KNN(n_neighbors=neighbors_num)
        knn_model_maker.fit(train_values, train_targets)
        test_prediction = knn_model_maker.predict(chosen_values)
        accuracy_score = accuracy(test_prediction, chosen_target)
        return accuracy_score

    accuracy_test = np.zeros(30)
    accuracy_train = np.zeros(30)
    split = .3
    data_train, data_test, target_train, target_test = tts(iris['data'], iris['target'],
                                                           test_size=split, random_state=random_state,
                                                           stratify=iris['target'])
    for i in (range(0, 30, 1)):
        accuracy_test[i] = accuracy_from_set(data_train, target_train, data_test, target_test, i + 1)
    for i in (range(0, 30, 1)):
        accuracy_train[i] = accuracy_from_set(data_train, target_train, data_train, target_train, i + 1)
    # A second set of curves is created for the inset to show whether there is greater closeness in accuracy with a 
    # larger or smaller test set partition percentage.
    accuracy_testB = np.zeros(30)
    accuracy_trainB = np.zeros(30)
    split = .08
    data_train, data_test, target_train, target_test = tts(iris['data'], iris['target'],
                                                           test_size=split, random_state=random_state,
                                                           stratify=iris['target'])
    for i in (range(0, 30, 1)):
        accuracy_testB[i] = accuracy_from_set(data_train, target_train, data_test, target_test, i + 1)
    for i in (range(0, 30, 1)):
        accuracy_trainB[i] = accuracy_from_set(data_train, target_train, data_train, target_train, i + 1)
    trace0 = go.Scatter(
        x=list(range(0, 30, 1)),
        y=accuracy_test,
        line=dict(shape='linear', color='DarkRed'),
        name='Test Set Accuracy'
    )
    trace1 = go.Scatter(x=list(range(0, 30, 1)),
                        y=accuracy_train,
                        line=dict(shape='linear', color='LightCoral'),
                        name='Training Set Accuracy'
                        )
    trace2 = go.Scatter(
        x=list(range(0, 30, 1)),
        y=accuracy_trainB,
        line=dict(shape='linear', color='Turquoise'),
        xaxis='x2', yaxis='y2',
        name='Training Set Accuracy-8% test'
    )
    trace3 = go.Scatter(x=list(range(0, 30, 1)),
                        y=accuracy_testB,
                        line=dict(shape='linear', color='DarkCyan'),
                        xaxis='x2', yaxis='y2',
                        name='Test Set Accuracy-8% test'
                        )
    data = [trace0, trace1, trace2, trace3]
    layout = go.Layout(
        xaxis=dict(
            title='Number of Neighbors Fit'
        ),
        yaxis=dict(
            title='Accuracy'
        ),
        xaxis2=dict(
            domain=[.6, .95],
            anchor='x2',
            side='top'
        ),
        yaxis2=dict(
            domain=[.6, .95],
            anchor='y2',
            side='right'
        ),
        title='Overfitting/ Underfitting for kNN <br> Iris Data Set ',
        titlefont=dict(size=26)
    )
    Fig = go.Figure(data=data, layout=layout)
    py.plot(Fig, filename='kNN_Test_Train_extra_test')
    plotly_overfit_underfit_curve_url = 'https://plot.ly/~DanWig/42/overfitting-underfitting-for-knn-iris-data-set/#/'

    # ------ Place code above here /\ /\ /\ ------

    return plotly_overfit_underfit_curve_url


# 10 points
def exercise05():
    '''
        Data set: Boston
        Load sklearn's Boston data into a DataFrame (only the data and feature_name as column names)
        Load sklearn's Boston target values into a separate DataFrame
        Return back the average of AGE, average of the target (median value of homes or MEDV), and the target as NumPy values 
    '''

    # ------ Place code below here \/ \/ \/ ------
    boston_df = pd.DataFrame(boston['data'])
    boston_df.columns = boston.feature_names
    average_age = boston_df.mean(axis=0)['AGE']
    medv_as_numpy_values = np.array(boston['target'])
    average_medv = medv_as_numpy_values.mean()
    # ------ Place code above here /\ /\ /\ ------

    return average_age, average_medv, medv_as_numpy_values


# 10 points
def exercise06():
    '''
        Data set: Boston
        In the Boston dataset, the feature PTRATIO refers to pupil teacher ratio.
        Using a matplotlib scatter plot, plot MEDV median value of homes as y-axis and PTRATIO as x-axis
        Return back PTRATIO as a NumPy array
    '''

    # ------ Place code below here \/ \/ \/ ------
    # Two steps are repeated from exercise05 and could be simplified if brought out of the separate framework or
    # called from the other function.
    boston_df = pd.DataFrame(boston['data'])
    boston_df.columns = boston.feature_names
    medv_as_numpy_values = np.array(boston['target'])
    plt.scatter(boston_df['PTRATIO'], medv_as_numpy_values, color='mediumslateblue')
    font = {'fontfamily': 'Serif'}
    plt.xlabel('Pupil Teacher Ratio', fontsize=20, color='slateblue', **font)
    plt.ylabel('Median House Value', fontsize=20, color='slateblue', **font)
    plt.show()
    X_ptratio = np.array(boston_df['PTRATIO'])
    # ------ Place code above here /\ /\ /\ ------

    return X_ptratio


# 20 points
def exercise07():
    '''
        Data set: Boston
        Create a regression model for MEDV / PTRATIO and display a chart showing the regression line using matplotlib
        with a backdrop of a scatter plot of MEDV and PTRATIO from exercise06
        Use np.linspace() to generate prediction X values from min to max PTRATIO
        Return back the regression prediction space and regression predicted values
        Make sure to labels axes appropriately
    '''

    # ------ Place code below here \/ \/ \/ ------
    # 7 lines are repeated from exercise06 and could be simplified if brought out of the separate framework or
    # called from the other function.
    boston_df = pd.DataFrame(boston['data'])
    boston_df.columns = boston.feature_names
    medv_as_numpy_values = np.array(boston['target'])
    plt.scatter(boston_df['PTRATIO'], medv_as_numpy_values, color='mediumslateblue')
    font = {'fontfamily': 'Serif'}
    plt.xlabel('Pupil Teacher Ratio', fontsize=20, color='slateblue', **font)
    plt.ylabel('Median House Value', fontsize=20, color='slateblue', **font)
    pt_ratio = np.array(boston_df['PTRATIO']).reshape(-1, 1)
    our_model = lm.LinearRegression().fit(X=pt_ratio, y=medv_as_numpy_values)
    prediction_space = np.array(np.linspace(12.4, 22.2, num=50)).reshape(-1, 1)
    reg_model = our_model.predict(prediction_space)
    plt.plot(prediction_space, reg_model, color='peru', linewidth=2.5)
    ax = plt.gca()
    ax.set_facecolor('seashell')
    plt.show()
    # ------ Place code above here /\ /\ /\ ------

    return reg_model, prediction_space


class TestAssignment7(unittest.TestCase):
    def test_exercise07(self):
        rm, ps = exercise07()
        self.assertEqual(len(rm), 50)
        self.assertEqual(len(ps), 50)

    def test_exercise06(self):
        ptr = exercise06()
        self.assertTrue(len(ptr), 506)

    def test_exercise05(self):
        aa, am, mnpy = exercise05()
        self.assertAlmostEqual(aa, 68.57, 2)
        self.assertAlmostEqual(am, 22.53, 2)
        self.assertTrue(len(mnpy), 506)

    def test_exercise04(self):
        print('Skipping EX4 tests')
        exercise04()

    def test_exercise03(self):
        score = exercise03(8, .25)
        self.assertAlmostEqual(exercise03(8, .3), .955, 2)
        self.assertAlmostEqual(exercise03(8, .25), .947, 2)

    def test_exercise02(self):
        pred = exercise02([[6.7, 3.1, 5.6, 2.4], [6.4, 1.8, 5.6, .2], [5.1, 3.8, 1.5, .3]])
        self.assertTrue('setosa' in pred)
        self.assertTrue('virginica' in pred)
        self.assertTrue('versicolor' in pred)
        self.assertEqual(len(pred), 3)

    def test_exercise01(self):
        df, tn = exercise01()
        self.assertEqual(df.shape, (5, 4))
        self.assertEqual(df.iloc[0, 1], 3.5)
        self.assertEqual(df.iloc[2, 3], .2)
        self.assertTrue('setosa' in tn)
        self.assertEqual(len(tn), 3)


if __name__ == '__main__':
    unittest.main()