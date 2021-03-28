import numpy as np
import os
import pickle
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", default=5, type=int, help="epochs")
parser.add_argument("--batch_size", default=750, type=int, help="batch size")

args = parser.parse_args([] if "__file__" not in globals() else None)
dist=5
epochs=args.epochs
batch_size=args.batch_size

counter=0

for i in range(1,9):
    file=open(f"nearest{i}/data.pickle", "rb")
    while True:
        try:
            x=pickle.load(file)
            counter+=1
        except:
            break
batches=counter//batch_size

dim=(dist*2+1)**2+4
loc=f"testnearest5/model"
if not os.path.isdir(loc):
    model=Sequential()
    model.add(Input(shape=(dim,)))
    model.add(Dense(dim+70,activation="relu"))
    model.add(Dense(dim+70,activation="relu"))
    model.add(Dense(dim+70,activation="relu"))
    model.add(Dense(6,activation="linear"))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005), metrics="acc")

else:
    model=load_model(loc)

class data_generator(Sequence):

    def __init__(self, first=1,batches=batches,batch_size=batch_size):
    
        
        self.file=open(f"nearest{first}/data.pickle", "rb")
        self.num=1  
        self.batches=batches 
        self.batch_size=batch_size

    def __len__(self):
        return self.batches

    def __getitem__(self,idx):
        xs=np.zeros((self.batch_size,1,dim))
        ys=np.zeros((self.batch_size,1,6))
    
        for i in range(self.batch_size):
            try:
                x,y=pickle.load(self.file)
                
            except:
                self.num+=1
                if self.num>8:
                    self.num=2
                self.file=open(f"nearest{self.num}/data.pickle", "rb")
                x,y=pickle.load(self.file) 
            xs[i]=x
            ys[i]=y
        return xs,ys

start=np.random.randint(2,9)
generator=data_generator(first=start)
validation=data_generator(first=1,batches=batches//8)

model.fit(generator,validation_data=validation, workers=1,epochs=epochs)
model.save(loc)


