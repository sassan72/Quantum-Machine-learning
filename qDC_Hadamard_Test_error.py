import numpy as np
import pennylane as qml
import sklearn.decomposition
from sklearn import metrics
import math
import pandas as pd
from qiskit import IBMQ # for run on IBMQ devices 





data1 = np.loadtxt("CERVICAL-F16-REC-ORIGINAL-1x.txt") # import pandas as pd

#df = pd.read_csv (r'Path where the CSV file is stored\File name.csv')

data2 = np.loadtxt("CERVICAL-F16-REC-ORIGINAL-1y.txt") #

data3 = np.loadtxt("CERVICAL-F16-REC-ORIGINAL-1x-test.txt") #

data4 = np.loadtxt("CERVICAL-F16-REC-ORIGINAL-1y-test.txt") #

X_train=data1[:]

y_train= data2[:,] 

y_test= data4[:,] 

X_test=data3[:]

X_train1 = sklearn.preprocessing.normalize(X_train, norm='l2',axis=0)

X_test1 = sklearn.preprocessing.normalize(X_test, norm='l2', axis=0)

X_train2 = sklearn.preprocessing.normalize(X_train1, norm='l2',axis=1)

X_test2 = sklearn.preprocessing.normalize(X_test1, norm='l2', axis=1)


x1_train=[]
x2_train=[]

for i in range(len(X_train2)):
    if y_train[i]==1:
        a=X_train2[i]
        x1_train.append(a)
    else:
        b=X_train2[i]
        x2_train.append(b)
        
feature_size = 8 # len(X_train[0])

num_qubits = 4 # log(feature_size) + 1(ancilla)

provider = IBMQ.enable_account('1b50b19485df6a5e93f18480d395e44d31f899d2bed6662a15d251ac75779a2d0ba039e1f1122c0ae024ba23fb2b295e551a721c0928edf8f3bd8f14283e8d43')

dev = qml.device('qiskit.ibmq', wires=num_qubits, backend='ibmq_quito', provider=provider, shots=8192) # for IBMQ devices

#dev = qml.device("default.qubit", wires=num_qubits, shots=8192) # for simulator

def ops(X):

    qml.templates.MottonenStatePreparation(X, wires=[1, 2, 3])

ops1 = qml.ctrl(ops, control=0)

@qml.qnode(dev)
def circuit(X1, X2):

    qml.Hadamard(wires=0)

    ops1(X1)
    
    qml.PauliX(wires=0)
    
    ops1(X2)
    
    qml.PauliX(wires=0)

    
    qml.Hadamard(wires=0)
    
    
    return qml.expval(qml.PauliZ(0))


D1_old = []

for x1 in x1_train:
    for x2 in X_test2:
        a = circuit(x1, x2)
        D1_old.append(a)

D1 = qml.kernels.mitigate_depolarizing_noise(D1_old, num_wires=num_qubits, method='split_channel')

D2_old = []

for x1 in x2_train:
    for x2 in X_test2:
        b = circuit(x1, x2)
        D2_old.append(b)
        
D2 = qml.kernels.mitigate_depolarizing_noise(D2_old, num_wires=num_qubits, method='split_channel')


D11 = np.reshape(D1, (len(x1_train), len(X_test)))
D22 = np.reshape(D2, (len(x2_train), len(X_test)))

d11 = np.max(D11, axis=0)
d22 = np.max(D22, axis=0)
m = len(X_test)

Prediction=[]
for i in range(m):
    if d11[i] >= d22[i]:
        Prediction.append(1)
    else:
        Prediction.append(0)
    
BB = metrics.confusion_matrix(y_test, Prediction, labels=[0,1])

print(Prediction)