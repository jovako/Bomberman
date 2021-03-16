import numpy as np
import os
from tensorflow.keras.models import load_model
from agent_code.fulldnq.transform import transformfield
try:
    from agent_code.fulldnq.train import action
except:
    print("failed")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



def setup(self):
    if not os.path.isdir("mymodel"):
        self.neednew=True
        print("neues model gebaut\n\n\n")
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
