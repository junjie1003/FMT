from keras import backend as K
import numpy as np
from keras.models import load_model, Model

per=0.05

def profiling(model,x_train):
    average_ats=np.zeros(19,)
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs] 
    for i in range(25000):
        print(i)
        test = x_train[i].reshape(-1,32,32,3)
        layer_outs = [func([test]) for func in functors]

        al=[]

        for l in layer_outs:
            out=l[0][0]
            tp=str(type(out[0]))[14:21]

            if tp=='float32':
                tmp=out
            else:   

                a=len(out)
                b=len(out[0])
                c=len(out[0][0])
                tmp=[]
                for i in range(c):
                    s=0
                    for j in range(a):
                        for k in range(b):
                            s+=out[j][k][i]
                    tmp.append(s)
                tmp = np.array(tmp)

            al.append(tmp)

        nl = np.array(al)
        if i==0:
            average_ats=nl
        else:
            average_ats=average_ats+nl

    average_ats=average_ats/25000
    np.save('alexnet_cifar10_average_ats.npy',average_ats)
    return average_ats

def forward_analysis(model,x_interest):
    interest_ats=[]
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs] 
    for i in range(len(x_interest)):
        print(i)
        test = x_interest[i].reshape(-1,32,32,3)
        layer_outs = [func([test]) for func in functors]

        al=[]

        for l in layer_outs:
            out=l[0][0]
            tp=str(type(out[0]))[14:21]

            if tp=='float32':
                tmp=out
            else:   

                a=len(out)
                b=len(out[0])
                c=len(out[0][0])
                tmp=[]
                for i in range(c):
                    s=0
                    for j in range(a):
                        for k in range(b):
                            s+=out[j][k][i]
                    tmp.append(s)
                tmp = np.array(tmp)
            al.append(tmp)

        nl = np.array(al)
        interest_ats.append(nl)

   
    return interest_ats

def find_min(l,num):
    l2=l.tolist()
    ind=[]
    Inf = 100
    for i in range(num):
        ind.append(l2.index(min(l2)))
        l2[l2.index(min(l2))]=Inf
    return ind

def backword_analysis(model,delta_y,label):
    contrib=[]
    cut=[]
    contrib.append(delta_y[18][label])
 #   print(contrib)
    layer_name=['conv2d_1','batch_normalization_1','conv2d_2','batch_normalization_2','conv2d_3','conv2d_4','conv2d_5','dense_1','dense_2','dense_3']
    weights=[]
    
    for l in layer_name:
        weight=model.get_layer(l).get_weights()#weights存在[0] bias存在[1]
        print(weight[0].shape)
        weights.append(weight[0])

    #dense_3
    local_contrib=np.zeros((2048,),dtype=float)
    tmp=np.zeros((2048,),dtype=float)
    print('dense_3')

    for j in range(2048):
        c=contrib[0]*delta_y[18][label]*weights[9][j][label]*delta_y[17][j]
     #   if c>0:
     #       local_contrib[j]+=1
     #   elif c<0:
     #       local_contrib[j]-=1
        local_contrib[j]+=c
    
    contrib.append(local_contrib)
    ind=find_min(local_contrib,int(per*2048))
    cut.append(ind)
    contrib[1][ind]=0

    #dense_2
    print('dense_2')
    local_contrib=np.zeros((2048,),dtype=float)
    tmp_cut=[]
    for i in range(2048):
        tmp_contri=np.zeros((2048,),dtype=float)
        for j in range(2048):
            c=contrib[1][i]*delta_y[16][i]*weights[8][j][i]*delta_y[15][j]
            if c>0:
                local_contrib[j]+=1
                tmp_contri[j]+=1
            elif c<0:
                local_contrib[j]-=1
                tmp_contri[j]-=1
        ind=find_min(tmp_contri,int(per*2048))
        tmp_cut.append(ind)
          #  local_contrib[j]+=c
    contrib.append(local_contrib)
    cut.append(tmp_cut)
    
    #dense_1
    print('dense_1')
    local_contrib=np.zeros((1024,),dtype=float)
    tmp_cut=[]
    for i in range(2048):
        tmp_contri=np.zeros((2048,),dtype=float)
        for j in range(1024):
            c=contrib[2][i]*delta_y[14][i]*weights[7][j][i]*delta_y[13][j]
            if c>0:
                local_contrib[j]+=1
                tmp_contri[j]+=1
            elif c<0:
                local_contrib[j]-=1
                tmp_contri[j]-=1
        ind=find_min(tmp_contri,int(per*2048))
        tmp_cut.append(ind)
          #  local_contrib[j]+=c
    contrib.append(local_contrib)
    cut.append(tmp_cut)
    
    #conv2d_5
    print('conv2d_5')
    print(contrib[3].shape)
    print(delta_y[11].shape)
    print(weights[6].shape)
    local_contrib=np.zeros((384,),dtype=float)
    tmp_cut=[]
    for i in range(256):
        tmp_contri=np.zeros((384,),dtype=float)
        for j in range(384):
            c=0
            for k in range(3):
                for l in range(3):  
                    c+=contrib[3][i]*delta_y[11][i]*weights[6][k][l][j][i]*delta_y[10][j]
            if c>0:
                local_contrib[j]+=1
                tmp_contri[j]+=1
            elif c<0:
                local_contrib[j]-=1
                tmp_contri[j]-=1
        ind=find_min(tmp_contri,int(per*384))
        tmp_cut.append(ind)
          #  local_contrib[j]+=c
    contrib.append(local_contrib)
    cut.append(tmp_cut)
    #conv2d_4
    print('conv2d_4')
    local_contrib=np.zeros((384,),dtype=float)
    tmp_cut=[]
    for i in range(384):
        tmp_contri=np.zeros((384,),dtype=float)
        for j in range(384):
            c=0
            for k in range(3):
                for l in range(3):
                    c+=contrib[4][i]*delta_y[10][i]*weights[5][k][l][j][i]*delta_y[9][j]
            if c>0:
                local_contrib[j]+=1
                tmp_contri[j]+=1
            elif c<0:
                local_contrib[j]-=1
                tmp_contri[j]-=1
        ind=find_min(tmp_contri,int(per*384))
        tmp_cut.append(ind)
            #local_contrib[j]+=c
    contrib.append(local_contrib)
    cut.append(tmp_cut)
    #conv2d_3
    print('conv2d_3')
    local_contrib=np.zeros((256,),dtype=float)
    tmp_cut=[]
    for i in range(384):
        tmp_contri=np.zeros((256,),dtype=float)
        for j in range(256):
            c=0
            for k in range(3):
                for l in range(3):
                    c+=contrib[5][i]*delta_y[9][i]*weights[4][k][l][j][i]*delta_y[8][j]
            if c>0:
                local_contrib[j]+=1
                tmp_contri[j]+=1
            elif c<0:
                local_contrib[j]-=1
                tmp_contri[j]-=1
        ind=find_min(tmp_contri,int(per*256))
        tmp_cut.append(ind)
          #  local_contrib[j]+=c
    contrib.append(local_contrib)
    cut.append(tmp_cut)
    #batch_normalization_2
    print('batch_normalization_2')
  #  local_contrib=np.zeros((256,),dtype=float)
  #  for i in range(256):
     #   c+=contrib[6][i]*delta_y[6][i]*weights[3][i]*delta_y[5][i]
      #  local_contrib[j]+=c
    contrib.append(contrib[6])
  #  cut.append([])

    #conv2d_2
    print('conv2d_2')
    local_contrib=np.zeros((96,),dtype=float)
    tmp_cut=[]
    for i in range(256):
        tmp_contri=np.zeros((96,),dtype=float)
        for j in range(96):
            c=0
            for k in range(3):
                for l in range(3):
                    c+=contrib[7][i]*delta_y[5][i]*weights[2][k][l][j][i]*delta_y[4][j]
            if c>0:
                local_contrib[j]+=1
                tmp_contri[j]+=1
            elif c<0:
                local_contrib[j]-=1
                tmp_contri[j]-=1
        ind=find_min(tmp_contri,int(per*96))
        tmp_cut.append(ind)
          #  local_contrib[j]+=c
    contrib.append(local_contrib)
    cut.append(tmp_cut)

    #batch_normalization_2
   # local_contrib=np.zeros((96,),dtype=float)
   # for i in range(96):
       # c+=contrib[8][i]*delta_y[2][i]*weights[1][i]*delta_y[1][i]
       # local_contrib[j]+=c
    contrib.append(contrib[8])

    #conv2d_1
    print('conv2d_1')
    print(len(contrib))
    print(delta_y[1].shape)
    print(weights[1].shape)
    print(delta_y[0].shape)
    local_contrib=np.zeros((3,),dtype=float)
    tmp_cut=[]
    for i in range(96):
        tmp_contri=np.zeros((3,),dtype=float)
        for j in range(3):
            c=0
            for k in range(3):
                for l in range(3):
                    c+=contrib[9][i]*delta_y[1][i]*weights[0][k][l][j][i]*delta_y[0][j]
            if c>0:
                local_contrib[j]+=1
                tmp_contri[j]+=1
            elif c<0:
                local_contrib[j]-=1
                tmp_contri[j]-=1
        ind=find_min(tmp_contri,int(per*3))
        tmp_cut.append(ind)
          #  local_contrib[j]+=c
    contrib.append(local_contrib)
    cut.append(tmp_cut)
 #   print(weight[1].shape)
    return contrib,cut
    
def pre_processing(x_train, x_validation):
    x_train = x_train.astype('float32')/255.0
    x_validation = x_validation.astype('float32')/255.0
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_validation -= x_train_mean

    return x_train, x_validation

if __name__ == "__main__":

    model=load_model("alexnet_model.200.h5")
    model.summary()
    x_train=np.load('/home/zhangyingyi/train/change/cifar10_x_train.npy')
    y_train=np.load('/home/zhangyingyi/train/change/cifar10_y_train.npy')
    x_validation=np.load('/home/zhangyingyi/train/change/cifar10_x_validation.npy')
    y_validation=np.load('/home/zhangyingyi/train/change/cifar10_y_validation.npy')
    x_train, x_validation=pre_processing(x_train, x_validation)
  #  x_interest=x_validation[:100]
   # average_ats=profiling(model,x_train)
    average_ats=np.load('alexnet_cifar10_average_ats.npy')
    for i in range(10):
        label=i
        x_interest=x_train[y_train.reshape(-1)==label]
        interest_ats=np.load('alexnet_cifar10_interest_ats_{}.npy'.format(label))
      #  interest_ats=forward_analysis(model,x_interest)
      #  np.save('alexnet_cifar10_interest_ats_{}.npy'.format(label),interest_ats)
        delta_y=[]
        for i in range(len(interest_ats)):
            delta_y.append(interest_ats[i]-average_ats)
        delta_y = np.array(delta_y)[0]
        np.save('alexnet_cifar10_delta_y_{}.npy'.format(label),delta_y)
        contrib,cut=backword_analysis(model,delta_y,label)
        print(cut)
        np.save('alexnet_cifar10_cut_{}.npy'.format(label),cut)
   # print(average_ats.shape)
   # print(interest_ats.shape)
 #   interest_ats=np.load('alexnet_cifar10_interest_ats_100.npy')
    