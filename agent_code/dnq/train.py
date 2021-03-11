from collections import deque
from typing import List
import events as e
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow import convert_to_tensor as ct

TRANSITION_HISTORY_SIZE = 20

def transformfield(game_state):
        if game_state==None:
            return None
        bombs=game_state["bombs"]
        me=game_state["self"]
        others=game_state["others"]
        bombfield=np.zeros((17,17))
        playerfield=np.zeros((17,17))
        for bomb in bombs:
            bombfield[bomb[0]]=bomb[1]
        playerfield[me[3]]=1
        fieldstate=game_state["field"]-bombfield-game_state["explosion_map"]
        for other in others:
            if other[2]==True:
                playerfield[other[3]]=-1
        features=np.array([fieldstate,playerfield])
        return ct(features.reshape(1,-1))

def buildnet():
    model=Sequential()
    model.add(Input(shape=(17*17*2,)))
    model.add(Dense(600,activation="relu"))
    model.add(Dense(600,activation="relu"))
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
        return 

def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.gamma=0.6
    self.temp=1
    self.epsilon=0.15
    if self.neednew==True:
        self.model=buildnet()
    self.target=buildnet()
    self.target.set_weights(self.model.get_weights())

def game_events_occurred(self, old_game_state: dict, self_action, new_game_state: dict, events: List[str]):
    self.transitions.append([old_game_state, self_action,new_game_state, reward_from_events(self, events)])
    
def end_of_round(self, last_game_state, last_action, events):
    self.transitions.append([last_game_state, last_action, None, reward_from_events(self, events)])
    if self.temp>0.01:
        self.temp=self.temp*0.99995
    
    for old,action,new,reward in self.transitions: 
        if old==None:
            continue 
        newfield=transformfield(new)
        oldfield=transformfield(old)
        action=strtoint(action)
        qpred=self.model.predict(oldfield)
        if new == None:
            qpred[0][action]=reward
        else:
            next=self.target.predict(newfield)
            qpred[0][action]=reward+self.gamma*np.amax(next)
        self.model.fit(oldfield,qpred,epochs=1,verbose=0)
    if last_game_state["round"]%10==0:
        self.model.save("mymodel")
    self.target.set_weights(self.model.get_weights())

    
def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 300,
        e.KILLED_SELF: -300,
        e.GOT_KILLED: -250,
        e.INVALID_ACTION: -100,
        e.WAITED: -50,
        e.CRATE_DESTROYED: 50
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum

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

