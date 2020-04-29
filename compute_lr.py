max_iter=240

gamma=0.9
interval=30
epoch=184
lr = 1e-04
for i in range(1,max_iter+1):

    if (i%(interval)==0 and i<max_iter):
        factor=(1-i/max_iter)**gamma
        lr=lr*factor

        print("current lr",lr,i*epoch)
