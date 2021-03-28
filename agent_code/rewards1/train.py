import events as e
import numpy as np
from transform import subfield as transformfield
from rewards import advancedrewards as getrewards
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
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
    model.add(Dense(125+50,activation="relu"))
    model.add(Dense(125+50,activation="relu"))
    model.add(Dense(6,activation="linear"))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics="acc")
    return model

def setup_training(self):
    self.oldfields=[]
    
    self.actions=[]
    self.allevents=[]
 
    if self.neednew:
        self.model=buildmodel(self)
    self.target=buildmodel(self)
    self.target.set_weights(self.model.get_weights())
        

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if not old_game_state==None:
        self.oldfields.append(transformfield(old_game_state,dist=5,nearest=True)[0])
        self.actions.append(strtoint(self_action))
        self.allevents.append(events)
   
        
def end_of_round(self, last_game_state, last_action, events):
    self.oldfields.append(transformfield(last_game_state,dist=5,nearest=True)[0])
    self.actions.append(strtoint(last_action))
    self.allevents.append(events)
    
    x=np.asarray(self.oldfields)
    rewards=getrewards(self.allevents)

    with open("data.pickle","ab") as file:
        for i in range(len(x)):
            act=self.actions[i]
            y=np.zeros(6)
            y[act]=rewards[i]
            pickle.dump((x[i],y),file)
        file.close()

    self.oldfields=[]
    self.actions=[]
    self.allevents=[]








            
