import numpy as np
import ast

input = open('census_model_fix1.txt', 'r')
lines = input.readlines()

print(len(lines))

for i in range(6):
    wline =  2 * i
    bline = 1 + 2 * i

    w = np.array(ast.literal_eval(lines[wline]))
    b = np.array(ast.literal_eval(lines[bline]))

    w = w.transpose(1,0)
    print(w.shape)

    wout = open('../weights/w_fix1_' + str(i + 1) + '.txt', 'w')
    bout = open('../bias/b_fix1_' + str(i + 1) + '.txt', 'w')

    wout.write(str(w.tolist()))
    bout.write(str(b.tolist()))

    wout.flush()
    bout.flush()

    wout.close()
    bout.close()

input.close()
