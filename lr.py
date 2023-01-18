import matplotlib.pyplot as plt
import numpy as np

data = 'data.csv'

my_data = np.genfromtxt(data, delimiter=',')
X = my_data[:,0].reshape(-1,1)
ones = np.ones([X.shape[0],1])
X = np.concatenate([ones, X],1)
y = my_data[:,1].reshape(-1,1)

plt.scatter(my_data[:,0].reshape(-1,1),y)
plt.show()
###