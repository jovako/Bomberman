import numpy as np
from tensorflow.keras.models import load_model
from transform import subfield as transformfield
import os

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



def setup(self):
    if not os.path.isdir("model"):
        self.neednew=True
    else:
        self.model=load_model("model")
        self.neednew=False
    
def act(self, game_state):
    field=transformfield(game_state,dist=4,nearest=True)
    p=self.model.predict(field)[0]
    if self.train:
        if np.random.rand()<self.epsilon:
            return np.random.choice(ACTIONS,p=[0.2,0.2,0.2,0.2,0.1,0.1])
    return ACTIONS[np.argmax(p)]
        
