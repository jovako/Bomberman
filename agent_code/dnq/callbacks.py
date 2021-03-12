import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow import convert_to_tensor as ct
try:
    from agent_code.dnq.train import action
except:
    pass

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def transformfield(game_state):
    if game_state==None:
        return None
    field=np.ones((7,7))
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

def setup(self):
    if not os.path.isdir("mymodel"):
        self.neednew=True
        print("neues model gebaut")
    else:
        self.neednew=False
        self.model=load_model("mymodel")
    
def act(self, game_state):
    field=transformfield(game_state)
    p=self.model.predict(field)[0]
    if self.train:
        return ACTIONS[action(self,p)]
    else:
        return ACTIONS[np.argmax(p)]
