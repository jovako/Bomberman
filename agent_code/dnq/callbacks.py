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
        return ct(features.reshape(1,-1))

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
