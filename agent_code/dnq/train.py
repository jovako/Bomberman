import events as e
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow import convert_to_tensor as ct


def transformfield(game_state):
    field=-np.ones((7,7)) 
    me=game_state["self"][3]
    xmin=max(me[0]-3,0)
    ymin=max(me[1]-3,0)
    xmax=min(me[0]+4,17)
    ymax=min(me[1]+4,17)
    fieldxmin=max(3-me[0],0)
    fieldymin=max(3-me[1],0)
    fieldxmax=min(20-me[0],7)
    fieldymax=min(20-me[1],7)
    bombs=game_state["bombs"]
    others=game_state["others"]
    newfield=np.zeros((17,17))
    for bomb in bombs:
        newfield[bomb[0]]=-bomb[1]
    for other in others:
        if other[2]==True:
            newfield[other[3]]=2
    field[fieldxmin:fieldxmax,fieldymin:fieldymax]=(game_state["field"]-game_state["explosion_map"]+newfield)[xmin:xmax,ymin:ymax]
    return field.reshape(1,-1)

def buildnet():
    model=Sequential()
    model.add(Input(shape=(49,)))
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
    self.epsilon=0.2
    if self.neednew==True:
        self.model=buildnet()
    self.target=buildnet()
    self.target.set_weights(self.model.get_weights())
    self.oldfields=[]
    self.newfields=[]
    self.actions=[]
    self.rewards=[]

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if not old_game_state==None:
        self.oldfields.append(transformfield(old_game_state))
        self.newfields.append(transformfield(new_game_state))
        self.actions.append(strtoint(self_action))
        self.rewards.append(reward_from_events(self,events))
    
def end_of_round(self, last_game_state, last_action, events):
    self.oldfields.append(transformfield(last_game_state))
    self.actions.append(strtoint(last_action))
    self.rewards.append(reward_from_events(self,events))
    oldfields=np.asarray(self.oldfields).reshape(-1,49)
    newfields=np.asarray(self.newfields).reshape(-1,49)
    rewards=np.asarray(self.rewards)
    actions=np.asarray(self.actions)
    qpred=self.model.predict(oldfields)
    next=self.target.predict(newfields)
    qpred[-1,actions[-1]]=rewards[-1]
    qpred[np.arange(len(qpred)-1),actions[:-1]]=rewards[:-1]+self.gamma*np.amax(next,axis=-1)
    self.model.fit(oldfields,qpred,verbose=0)
    if last_game_state["round"]%10==0:
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
        return np.random.choice([0,1,2,3,4,5], p=[0.2,0.2,0.2,0.2,0.1,0.1])
    else:
        return np.argmax(p)

