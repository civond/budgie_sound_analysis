import numpy as np
d = [0.000, 0.166, 0.333]   #ideal target distances, these can be all zeros.
p = [0.000, 0.254, 0.998]   #your performance goes here

print("d is: " + str(["%.8f" % elem for elem in d]))
print("p is: " + str(["%.8f" % elem for elem in p]))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val = rmse(np.array(d), np.array(p))
print("rms error is: " + str(rmse_val))