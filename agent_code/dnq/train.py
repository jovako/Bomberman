from agent_code.dnq.definition import strtoint, buildnet, transformfield, np, deque, namedtuple, List, e

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 20

def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.gamma=0.6
    self.temp=1
    self.epsilon=0.15
    if self.neednew==True:
        self.model=buildnet()
    self.target=buildnet()
    self.target.set_weights(self.model.get_weights())

def game_events_occurred(self, old_game_state: dict, self_action, new_game_state: dict, events: List[str]):
    self.transitions.append([old_game_state, self_action,new_game_state, reward_from_events(self, events)])
    
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.transitions.append([last_game_state, last_action, None, reward_from_events(self, events)])
    if self.temp>0.01:
        self.temp=self.temp*0.99995
    
    for old,action,new,reward in self.transitions: 
        if old==None:
            continue 
        newfield=transformfield(new)
        oldfield=transformfield(old)
        action=strtoint(action)
        qpred=self.model.predict(oldfield)
        if new == None:
            qpred[0][action]=reward
        else:
            next=self.target.predict(newfield)
            qpred[0][action]=reward+self.gamma*np.amax(next)
        self.model.fit(oldfield,qpred,epochs=1,verbose=0)
    if last_game_state["round"]%10==0:
        self.model.save("mymodel")
    self.target.set_weights(self.model.get_weights())

    
def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 300,
        e.KILLED_SELF: -300,
        e.GOT_KILLED: -250,
        e.INVALID_ACTION: -100,
        e.WAITED: -50,
        e.CRATE_DESTROYED: 50
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum


