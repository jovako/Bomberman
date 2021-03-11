from agent_code.dnq.definition import transformfield, load_model, action, np, os


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
