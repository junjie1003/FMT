import numpy as np
import ast
from scipy.stats import norm
#
#input = open('credit.txt', 'r')
#lines = input.readlines()



#print(len(lines))
y = []


data = []
dist = []

no_attr = 401

#for line in lines:
#    #array.append([int(x) for x in line.split(",")])
#
#    array = [float(x) for x in line.split(",")]
#    x = array[:(len(array) - 1)]
#    data.append(np.array(x))
#    no_attr = len(x)
#    #np.append(data, np.array(x).reshape(-1,20), axis=0)
#    #output_x0 = open("data{}.txt".format(i), 'w+')
#    #output_x0.write(x)
#
#    #output_x0.write("{}\n".format(x))
#
#    #output_x0.close()
#
#    #y.append(array[-1])
#
#    i = i + 1
#
#input.close()
#
#data = np.array(data)
#print(data)
#print(i)
#print(no_attr)
data=np.load('compas_x_train.npy')
print(data.shape)
min_x=np.zeros(no_attr,)
max_x=np.zeros(no_attr,)
for i in range(no_attr):
    min_x[i]=10000

for i in range(4316):
    for j in range(no_attr):
        if data[i][j]<min_x[j]:
            min_x[j]=data[i][j]
        if data[i][j]>max_x[j]:
            max_x[j]=data[i][j]
s=''
for i in range(no_attr):
    s+='('+str(int(min_x[i]))+','+str(int(max_x[i]))+'),'

print(s)

output_y = open('distribution_processed.txt', 'w+')
mu_ = []
std_ = []
for i in range(0, no_attr):
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data[:, i])
    to_add = []
    to_add.append(mu)
    to_add.append(std)
    dist.append(to_add)
    output_y.write("{}\n".format(to_add))
    mu_.append(mu)
    std_.append(std)

output = open('distribution_sep_processed.txt', 'w+')
output.write("{}\n".format(mu_))
output.write("{}\n".format(std_))
output.close()
output_y.close()
