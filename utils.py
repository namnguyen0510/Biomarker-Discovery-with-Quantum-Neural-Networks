import numpy as np

# NORMALIZE INPUT
def normalize(x):
	x = (x-x.min())/(x.max()-x.min())
	return x

# COMPUTE MODULO OF COMPLEX NUMBER - PROBABILITY OF RETURED STATE
def compute_modulo(p):
    re = p.real
    img = p.imag
    p = re**2 + img**2
    return np.sqrt(p)

# THRESHOLDING PROBABILITY VECTOR
def threshold(p, t):
    p[p >= t] = 1
    p[p < t] = 0
    return p

# GET PROBABILITY OF TARGET VARIABLES
def get_y_prob(y):
    prob = []
    for i in range(len(y.columns)):
        t = y.iloc[:,i]
        p = np.zeros(y.iloc[:,i].shape)
        x, c = np.unique(y.iloc[:,i], return_counts = True)
        p[t == 0] = c[0]/len(t)
        p[t == 1] = c[1]/len(t)
        prob.append(p)
    prob = np.array(prob).reshape(len(y),len(y.columns))
    return prob
# VISUALIZATION
def improve_text_position(x):
	positions = ['top center', 'bottom center']
	return [positions[i % len(positions)] for i in range(len(x))]
