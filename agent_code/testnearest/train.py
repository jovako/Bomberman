import events as e
import numpy as np
from transform import subfield as transformfield
from rewards import advancedrewards as getrewards
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import pickle

ACTIONS = np.asarray(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])


def strtoint(action):
    if not action in ACTIONS:
        return 4
    else:
        return np.arange(6)[ACTIONS==action][0]
    
def buildmodel(self):
    model=Sequential()
    model.add(Input(shape=(125,)))
    model.add(Dense(200,activation="relu"))
    model.add(Dense(200,activation="relu"))
    model.add(Dense(200,activation="relu"))
    model.add(Dense(6,activation="linear"))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics="acc")
    return model

def setup_training(self):
    self.oldfields=[]
    self.newfields=[]
    self.actions=[]
    self.allevents=[]
    self.alpha=0.3
    self.gamma=0.9
    self.epsilon=pickle.load(open("epsilon","rb"))
    print(self.epsilon)
    self.mini=200
    self.batch_size=32
    if self.neednew:
        self.model=buildmodel(self)
    self.target=buildmodel(self)
    self.target.set_weights(self.model.get_weights())
        

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if not old_game_state==None:
        self.oldfields.append(transformfield(old_game_state,dist=5,nearest=True)[0])
        self.newfields.append(transformfield(new_game_state,dist=5,nearest=True)[0])
        self.actions.append(strtoint(self_action))
        self.allevents.append(events)
   
        
def end_of_round(self, last_game_state, last_action, events):
    self.oldfields.append(transformfield(last_game_state,dist=5,nearest=True)[0])
    self.actions.append(strtoint(last_action))
    self.allevents.append(events)
    
    x=np.asarray(self.oldfields)
    next=np.asarray(self.newfields)
    y=self.model.predict(x)
    target=self.target.predict(next)
    rewards=getrewards(self.allevents)
    
    for i in range(len(x)-1):
        act=self.actions[i]
        y[i,act]+=self.alpha*(rewards[i]+self.gamma*max(target[i])-y[i,act])
    y[-1,self.actions[-1]]+=self.alpha*(rewards[-1]-y[i,self.actions[-1]])

    mini=min(self.mini,len(x))
    minibatch=np.random.permutation(len(x))[:mini]
    self.model.fit(x[minibatch],y[minibatch],batch_size=self.batch_size)
    if last_game_state["round"]%20==0:
        with open("rewards","ab") as file:
            pickle.dump(np.mean(rewards),file)
            file.close()
        
    if last_game_state["round"]%200==0:
        self.model.save("model")
        self.target.set_weights(self.model.get_weights())
        self.epsilon=max(0.01,self.epsilon*0.998)
        with open("epsilon","wb") as file:
            pickle.dump(self.epsilon,file)
            file.close()
    self.oldfields=[]
    self.newfields=[]
    self.actions=[]
    self.allevents=[]








            
