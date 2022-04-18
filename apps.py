# phaly to sari important libarries import karni hain

from ast import Import
import imp
from random import random
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt    # 'pyyplot':Unknownwords
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm  import SVC
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import accuracy_score

# app ki heading?
st.write('''
         # Exploratory diffeence models and datasets
         Daiktey hain kon kon best ha in may sa?
         ''')

# dataset ka name ak box may daal ka sidebaar pay laga do

datasets_name = st.sidebar.selectbox(
    'select Dataset',
    ('iris', 'Breast  Cancer', 'wine')
)


# claassifier ka name ak box may daal ka sidebaar pay laga do

classifier_name = st.sidebar.selectbox(
    'select classifier',
    ('KNN', 'SVM', 'Random Forest')
)

#ab ham aik function define karin ga or dataset ko load karyen ga

def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data= datasets.load_iris()
    elif dataset_name == 'wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target 
    return x,y

# ab is function ko bula layn ga or X , y variable ka equal rakh layen ga
X , y = get_dataset(datasets_name)

# ab haum apny dataset ki shape ko app pay prient karin ga
st.write('Shape of dataset:', X.shape)
st.write('Number of classes:', len(np.unique(y)))

def add_parameter_ui(classifier_name):
    params = dict() # create an empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C',0.01,10.0)
        params['C'] = C  #its the number of nearest_naibour
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K',1,15)
        params['K'] = K 
    else:
        max_depth = st.sidebar.slider('max_depth',2,15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators'] = n_estimators # number of trees
    return params

#ab 

params = add_parameter_ui(classifier_name)


def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf =clf= RandomForestClassifier(n_estimators = params['n_estimators'],
                                                        max_depth=params['max_depth'],random_state=1234)
    return clf
    
clf = get_classifier(classifier_name,params) 

# train and test
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

#ab hum nay apnay classifier ko training karni ha
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#model ka accuracy score check karin
acc = accuracy_score(y_test,y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = ',acc)

# plot data sets
# ab hum apny sary sary features ko 2 dimiensioal plot pat draw kar dayen fgay
pca = PCA(2)
X_projected =pca.fit_transform(X)

#ab hum apna data 0 or 1 dimenssion may slice kaar dayen
x1 = X_projected[:, 0]
x2 = X_projected[:, 0]

fig = plt.figure()
plt.scatter(x1,x2,
            c=y,alpha=0.8,
            cmap='viridis')


plt.xlabel('Principal Components 1')
plt.ylabel('Principal Components 2')
plt.colorbar()

#plt.show
st.pyplot(fig)

