import numpy as np
import os
from tensorflow.keras.models import load_model
try:
    from agent_code.dnq.train import action
except:
    print("failed")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def transformfield(self,game_state):
    dist=self.dist
    field=-np.ones((2*dist+1,2*dist+1))
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

def setup(self):
    self.dist=4
    self.dim=(self.dist*2+1)**2
    if not os.path.isdir("mymodel"):
        self.neednew=True
        print("neues model gebaut\n\n\n")
    else:
        self.neednew=False
        self.model=load_model("mymodel")
    
def act(self, game_state):
    field=transformfield(self,game_state)
    p=self.model.predict(field)[0]
    if self.train:
        return ACTIONS[action(self,p)]
    else:
        return ACTIONS[np.argmax(p)]
