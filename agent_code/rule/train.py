import events as e
import numpy as np
from transform import transformfield
import pickle

ACTIONS = np.asarray(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])

def strtohot(action):
    return np.asarray(ACTIONS==action,dtype=int)

def setup_training(self):
    self.oldfields=[]
    self.actions=[]
    self.allevents=[]

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if not old_game_state==None:
        self.oldfields.append(transformfield(old_game_state,dist=5,nearest=True))
        self.actions.append(strtohot(self_action))
        self.allevents.append(events)
    if e.KILLED_OPPONENT in events:
        print(events)
        print(self.allevents[-5])
        print(self.actions[-5])
        
def end_of_round(self, last_game_state, last_action, events):
    self.oldfields.append(transformfield(last_game_state,dist=5,nearest=True))
    self.actions.append(strtohot(last_action))
    self.allevents.append(events)
    weights=weights(self.allevents)
    model.fit(self.oldfields,self.actions,weights)
    model.save("model")

def weights(allevents):
    weights=np.ones(len(allevents))
    for i in range(len(allevents)):
        events=allevents[i]
        if e.COIN_COLLECTED in events:
            weights[i]+=1
        if e.INVALID_ACTION in events:
            weights[i]+=-5
        if e.KILLED_OPPONENT in events:
            weights[i-4]+=10
            weights[i-3]+=5
            weights[i-2]+=4
            weights[i-1]+=3
            weights[i]+=2
            print(allevents[i-4])
        if e.KILLED_SELF in events:
            weights[i-3]+=-300
            weights[i-2]+=-400
            weights[i-1]+=-400
            weights[i]+=-400
        if e.GOT_KILLED in events:
            weights[i-3]+=-200
            weights[i-2]+=-300
            weights[i-1]+=-300
            weights[i]+=-400
                
    weights+=abs(np.min(weights))
    weights/=abs(np.max(weights))) #normalize
    return weights







            
