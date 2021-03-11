import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def transformfield(game_state):
        if game_state==None:
            return None
        field=game_state["field"]
        explosions=game_state["explosion_map"]
        bombs=game_state["bombs"]
        me=game_state["self"]
        others=game_state["others"]
        bombfield=np.zeros((17,17))
        playerfield=np.zeros((17,17))
        for bomb in bombs:
            bombfield[bomb[0]]=bomb[1]
        playerfield[me[3]]=1
        fieldstate=field-bombfield-explosions
        for other in others:
            if other[2]==True:
                playerfield[other[3]]=-1
        features=np.array([fieldstate,playerfield])
        return tf.convert_to_tensor(features.reshape(1,-1))

def action(self,p):
    if self.epsilon>np.random.rand():
        prob=np.exp(self.temp*p)
        if np.sum(prob)==0. :
            prob=[0.2,0.2,0.2,0.2,0.1,0.1]
        else:
            prob=prob/np.sum(prob)
        return np.random.choice([0,1,2,3,4,5], p=prob)
    else:
        return np.argmax(p)

def buildnet():
    model=Sequential()
    model.add(Input(shape=(17*17*2,)))
    model.add(Dense(50,activation="relu"))
    model.add(Dense(50,activation="relu"))
    model.add(Dense(6,activation="linear"))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
    return model

def strtoint(action):
    if action=="UP":
        return 0
    if action=="RIGHT":
        return 1
    if action=="Down":
        return 2
    if action=="LEFT":
        return 3
    if action=="WAIT":
        return 4
    if action=="BOMB":
        return 5



