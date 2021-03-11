import os
import numpy as np
from tensorflow.keras.models import load_model
from agent_code.dnq.definition import action, transformfield

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
   
    if not os.path.isdir("mymodel"):
        self.neednew=True
        print("neues model gebaut")
    else:
        self.neednew=False
        self.model=load_model("mymodel")
    
def act(self, game_state: dict) -> str:
    field=transformfield(game_state)
    p=self.model.predict(field)[0]
    if self.train:
        return ACTIONS[action(self,p)]
    print(ACTIONS[np.argmax(p)])
    return ACTIONS[np.argmax(p)]
