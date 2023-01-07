import numpy as np
import ast


#input = open('credit.txt', 'r')
#lines = input.readlines()
#
#
#
#print(len(lines))
#y = []
#i = 0
#for line in lines:
#    #array.append([int(x) for x in line.split(",")])
#
#    array = [float(x) for x in line.split(",")]
#    x = array[:(len(array) - 1)]
#
#    output_x0 = open("data{}.txt".format(i), 'w+')
#    #output_x0.write(x)
#
#    output_x0.write("{}\n".format(x))
#
#    output_x0.close()
#
#    y.append(array[-1])
#
#    i = i + 1

data=np.load('compas_x_test.npy')
y=np.load('compas_y_test.npy')
print(data.shape)
print(y)
y1=[]
for i in range(1851):
    y1.append(np.argmax(y[i]))
print(y1)
for i in range(1851):
    x=[]
    for j in range(401):
        x.append(int(data[i][j]))
    output_x0 = open("data{}.txt".format(i), 'w+')
    #output_x0.write(x)

    output_x0.write("{}\n".format(x))

    output_x0.close()
    
output_y = open('labels.txt', 'w+')
output_y.write("{}\n".format(y1))

output_y.close()

