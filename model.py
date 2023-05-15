import pennylane as qml
import torch
import numpy as np


q = 11
dev = qml.device("default.qubit", wires=q)
#dev = qml.device("qiskit.aer", wires=q)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@qml.qnode(dev, interface="torch")
def quantum_sampler(a,b,n):
	t = torch.tensor(np.linspace(0,np.pi, n))
	for i in range(q):
		qml.Hadamard(wires=i)
	for k, dt in enumerate(t):
		for i in range(q):
			qml.RZ(dt,wires=i)
			qml.RY(a[k,i],wires=i)
			qml.RX(b[k,i],wires=i)
		for i in range(0, q - 1, 2):
			qml.CNOT(wires=[i, i + 1])
		for i in range(1, q - 1, 2):
			qml.CNOT(wires=[i, i + 1])
		qml.CNOT(wires = [q-1,0])
	return qml.state()
