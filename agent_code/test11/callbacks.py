import numpy as np
from tensorflow.keras.models import load_model
from transform import subfield as transformfield


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



def setup(self):
    self.model=load_model("model")
    
def act(self, game_state):
    field=transformfield(game_state,11).reshape(1,-1)
    p=self.model.predict(field)[0]
    return ACTIONS[np.argmax(p)]
