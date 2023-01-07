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

data=np.load('../distribution/german_x_train.npy')
y=np.load('../distribution/german_y_train.npy')
print(y)
y1=[]
for i in range(700):
    y1.append(np.argmax(y[i]))
print(y1)
for i in range(700):
    x=[]
    for j in range(58):
        x.append(int(data[i][j]))
    output_x0 = open("data{}.txt".format(i), 'w+')
    #output_x0.write(x)

    output_x0.write("{}\n".format(x))

    output_x0.close()
    
output_y = open('labels.txt', 'w+')
output_y.write("{}\n".format(y1))

output_y.close()

