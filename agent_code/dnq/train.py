import events as e
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow import convert_to_tensor as ct

def transformfield(self,game_state):
    dist=self.dist
    field=np.ones((2*dist+1,2*dist+1))
    me=game_state["self"][3]
    xmin=max(me[0]-dist,0)
    ymin=max(me[1]-dist,0)
    xmax=min(me[0]+dist+1,17)
    ymax=min(me[1]+dist+1,17)
    fieldxmin=max(dist-me[0],0)
    fieldymin=max(dist-me[1],0)
    fieldxmax=min(17+dist-me[0],2*dist+1)
    fieldymax=min(17+dist-me[1],2*dist+1)
    bombs=game_state["bombs"]
    others=game_state["others"]
    newfield=np.zeros((17,17))
    newfield[tuple(zip(*game_state["coins"]))]=4
    for bomb in bombs:
        newfield[bomb[0]]=-5+bomb[1]
    for other in others:
        if other[2]==True:
            newfield[other[3]]=2
    field[fieldxmin:fieldxmax,fieldymin:fieldymax]=(game_state["field"]+newfield)[xmin:xmax,ymin:ymax]
    return field.reshape(1,-1)

def buildnet(self):
    model=Sequential()
    model.add(Input(shape=(self.dim,)))
    model.add(Dense(100,activation="relu"))
    model.add(Dense(100,activation="relu"))
    model.add(Dense(6,activation="linear"))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
    return model

def strtoint(action):
    if action=="UP":
        return 0
    if action=="RIGHT":
        return 1
    if action=="DOWN":
        return 2
    if action=="LEFT":
        return 3
    if action=="WAIT":
        return 4
    if action=="BOMB":
        return 5

def setup_training(self):
    self.gamma=0.6
    self.epsilon=0.15
    if self.neednew==True:
        self.model=buildnet(self)
    self.target=buildnet(self)
    self.target.set_weights(self.model.get_weights())
    self.oldfields=[]
    self.newfields=[]
    self.actions=[]
    self.rewards=[]

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if not old_game_state==None:
        self.oldfields.append(transformfield(self,old_game_state))
        self.newfields.append(transformfield(self,new_game_state))
        self.actions.append(strtoint(self_action))
        self.rewards.append(reward_from_events(self,events))
    
def end_of_round(self, last_game_state, last_action, events):
    self.oldfields.append(transformfield(self,last_game_state))
    self.actions.append(strtoint(last_action))
    self.rewards.append(reward_from_events(self,events))
    oldfields=np.asarray(self.oldfields).reshape(-1,self.dim)
    newfields=np.asarray(self.newfields).reshape(-1,self.dim)
    rewards=np.asarray(self.rewards)
    actions=np.asarray(self.actions)
    qpred=self.model.predict(oldfields)
    next=self.target.predict(newfields)
    qpred[-1,actions[-1]]=rewards[-1]
    qpred[np.arange(len(qpred)-1),actions[:-1]]=rewards[:-1]+self.gamma*np.amax(next,axis=-1)
    self.model.fit(oldfields,qpred,verbose=0)
    if last_game_state["round"]%10==0:
        print(np.sum(rewards))
        self.model.save("mymodel")
        self.target.set_weights(self.model.get_weights())
    self.oldfields=[]
    self.newfields=[]
    self.actions=[]
    self.rewards=[]
    
def reward_from_events(self, events):
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 300,
        e.KILLED_SELF: -10000,
        e.GOT_KILLED: -250,
        e.INVALID_ACTION: -10,
        e.WAITED: -5,
        e.CRATE_DESTROYED: 50
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum

def action(self,p):
    if self.epsilon>np.random.rand():
        return np.random.choice([0,1,2,3,4,5], p=[0.25,0.25,0.25,0.25,0,0])
    else:
        return np.argmax(p)

