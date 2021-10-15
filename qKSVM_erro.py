import numpy as np
from sklearn.svm import SVC
import pennylane as qml
import sklearn.decomposition
from sklearn import metrics
from qiskit import IBMQ # for run on IBMQ devices 
import pandas as pd

pi = np.pi


data1 = pd.read_csv(r'Path where the CSV file is stored\File TDS.csv') 

data2 = pd.read_csv(r'Path where the CSV file is stored\File TLD.csv') 

data3 = pd.read_csv(r'Path where the CSV file is stored\File VDS.csv') 

data4 = pd.read_csv(r'Path where the CSV file is stored\File VLD.csv') 

y_train= data2[:,] 

X_train2=data1[:]

y_test= data4[:,] 

X_test2=data3[:]


X_train1 = sklearn.preprocessing.normalize(X_train2, norm='l2',axis=0)

X_test1 = sklearn.preprocessing.normalize(X_test2, norm='l2', axis=0)

X_train = sklearn.preprocessing.normalize(X_train1, norm='l2',axis=1)

X_test = sklearn.preprocessing.normalize(X_test1, norm='l2', axis=1)


feature_size = 8 # len(X_train[0])


n_qubits = 3 # log(feature_size) 

provider = IBMQ.enable_account('API_KEY')


dev = qml.device('qiskit.ibmq', wires=n_qubits, backend='IBMQ_DEVICE', provider=provider, shots=8192) # for IBMQ devices


#dev = qml.device("default.qubit", wires=n_qubits, shots=8192) # simulator


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

Σ12 = kernel_matrix(X1, X2)

Σ12_new = qml.kernels.mitigate_depolarizing_noise(Σ12, num_wires=n_qubits, method='split_channel')

svm = SVC(kernel=Σ12_new, class_weight='balanced').fit(X_train, y_train)



predictions = svm.predict(X_test)

print(predictions)

BB = metrics.confusion_matrix(y_test, predictions, labels=[0,1])

print(BB)