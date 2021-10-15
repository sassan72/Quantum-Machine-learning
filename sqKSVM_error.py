import numpy as np
import pennylane as qml
import sklearn.decomposition
from sklearn import metrics
import math
import pandas as pd

from qiskit import IBMQ # for run on IBMQ devices 




data1 = pd.read_csv(r'Path where the CSV file is stored\File TDS.csv') 

data2 = pd.read_csv(r'Path where the CSV file is stored\File TLD.csv') 

data3 = pd.read_csv(r'Path where the CSV file is stored\File VDS.csv') 

data4 = pd.read_csv(r'Path where the CSV file is stored\File VLD.csv') 

X_train2=data1[:]

y_train= data2[:,] 

y_test= data4[:,] 

X_test2=data3[:]

X_train1 = sklearn.preprocessing.normalize(X_train2, norm='l2',axis=0)

X_test1 = sklearn.preprocessing.normalize(X_test2, norm='l2', axis=0)

X_train = sklearn.preprocessing.normalize(X_train1, norm='l2',axis=1)

X_test = sklearn.preprocessing.normalize(X_test1, norm='l2', axis=1)



y_train_new =[]


for i in range(len(X_train)):
    if y_train[i]==1:
        a=0.67    # (1-imbalance_ratio)*y_train[i]
        y_train_new.append(a)
    else:
        b=-0.33   # imbalance_ratio*(-1)
        y_train_new.append(b)

feature_size = 8 # len(X_train[0])

n_qubits = 3 # log(feature_size)

pi = np.pi

provider = IBMQ.enable_account('5292b2d1663ccf6bcba1ee4fd995daf8f67b8f61b7c8290bd5fcc04ea411e0e9082e336fd95eab0c8f17f5d159c3592fe7be166a428becb8d44ba0ffb6d6be41')

dev = qml.device('qiskit.ibmq', wires=n_qubits, backend='ibmq_belem', provider=provider, shots=8192) # for IBMQ devices

#dev = qml.device("default.qubit", wires=n_qubits, shots=8192) # for simulator


projector = np.zeros((2**n_qubits, 2**n_qubits))

projector[0, 0] = 1

@qml.qnode(dev)
def kernel(x1, x2):
    """The quantum kernel."""
    qml.templates.MottonenStatePreparation(x1, wires=range(n_qubits))
    qml.inv(qml.templates.MottonenStatePreparation(x2, wires=range(n_qubits)))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))


def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])




def GP(X1, y1, X2):

    Σ12 = kernel_matrix(X1, X2)

    Σ12_new = qml.kernels.mitigate_depolarizing_noise(Σ12, num_wires=n_qubits, method='split_channel')
   
    #Σ12 = kernel_matrix(X1, X2) #B
    

    μ =  y1 @ Σ12_new 

    

    return μ


μ = GP(X_train, y_train_new, X_test)


C = [] #prediction

for i in μ:

    if i>= 0:
        C.append(1)

    else:
        C.append(0)


print(C)

BB = metrics.confusion_matrix(y_test, C, labels=[0,1])


print(BB)

