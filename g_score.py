import numpy as np
from numpy.linalg import matrix_rank
import pennylane as qml
import sklearn.decomposition
from sklearn import metrics
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.linalg import sqrtm
from scipy.linalg.interpolative import estimate_spectral_norm

# this code should be run on classical computer

data1 = pd.read_csv(r'Path where the CSV file is stored\File TDS.csv') 

data2 = pd.read_csv(r'Path where the CSV file is stored\File TLD.csv') 

data3 = pd.read_csv(r'Path where the CSV file is stored\File VDS.csv') 

data4 = pd.read_csv(r'Path where the CSV file is stored\File VLD.csv') 

y_train= data2[:,] 

X_train2=data1[:]

y_test= data4[:,] 

X_test2=data3[:]


pi = np.pi


X_train1 = sklearn.preprocessing.normalize(X_train2, norm='l2',axis=0)

X_test1 = sklearn.preprocessing.normalize(X_test2, norm='l2', axis=0)

X_train = sklearn.preprocessing.normalize(X_train1, norm='l2',axis=1)

X_test = sklearn.preprocessing.normalize(X_test1, norm='l2', axis=1)

feature_size = 8 # len(X_train[0])


n_qubits = 3 # log(feature_size) 

dev_kernel = qml.device("default.qubit", wires=n_qubits)


projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    qml.templates.MottonenStatePreparation(x1, wires=range(n_qubits))
    qml.inv(qml.templates.MottonenStatePreparation(x2, wires=range(n_qubits)))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))


def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])


K2 = kernel_matrix(X_train, X_train)


K1 = linear_kernel(X_train, X_train)


K22 = sqrtm(K2)

K11 = np.linalg.pinv(K1)

KK = (np.real(K22))@ K11 @ (np.real(K22)) # g(K^C||K^Q)


NORM = estimate_spectral_norm(KK, its=20)

print(np.sqrt(NORM))
