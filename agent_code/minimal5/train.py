import events as e
import numpy as np
from transform import minimal as transformfield
import pickle

ACTIONS = np.asarray(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])

def strtohot(action):
    return np.asarray(ACTIONS==action,dtype=int)

def setup_training(self):
    self.oldfields=[]
    self.actions=[]

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if not old_game_state==None:
        self.oldfields.append(transformfield(old_game_state))
        self.actions.append(strtohot(self_action))
    
def end_of_round(self, last_game_state, last_action, events):
    self.oldfields.append(transformfield(last_game_state))
    self.actions.append(strtohot(last_action))
    result=last_game_state["self"][1]
    res=[]
    for other in last_game_state["others"]:
        res.append(other[1])
    if len(res)==0:
        res=[0,0]
    if result>=max(res):
        for i in range(len(self.oldfields)):
            with open("data.pickle","ab") as file:
                pickle.dump((self.oldfields[i],self.actions[i]),file)
    self.oldfields=[]
    self.actions=[]
    
    

