import numpy as np
from tensorflow.keras.models import load_model
from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    self.model=load_model("model")
    self.last_actions=[]
    
def act(self, game_state):
    field=transformfield(game_state,5,nearest=True).reshape(1,-1)
    p=self.model.predict(field)[0]
    action=ACTIONS[np.argmax(p)]
    self.last_actions.append(action)
    if len(self.last_actions)>5: #simple loop-breaker
        if self.last_actions[-5:]==["UP","DOWN","UP","DOWN","UP"] or self.last_actions[-5:]==["DOWN","UP","DOWN","UP","DOWN"] or self.last_actions[-5:]==["RIGHT","LEFT","RIGHT","LEFT","RIGHT"] or self.last_actions[-5:]==["LEFT","RIGHT","LEFT","RIGHT","LEFT"] or self.last_actions[-4:]==["WAIT","WAIT","WAIT","WAIT"]:
            self.last_actions=[]
            return "BOMB"

    return ACTIONS[np.argmax(p)]

def transformfield(game_state,dist,nearest): 
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
    coins=game_state["coins"]
    for coin in coins:
        newfield[coin]=10
    for other in others:
        newfield[other[3]]=5
        coins.append(other[3])
    for bomb in bombs:
        newfield[bomb[0]]=-5+bomb[1]
    fieldmap=game_state["field"]+newfield
    field[fieldxmin:fieldxmax,fieldymin:fieldymax]=fieldmap[xmin:xmax,ymin:ymax]
    if not nearest:
        return field
    nearest=look_for_targets(fieldmap, me, coins)
    return np.concatenate((field,nearest),axis=None)

def look_for_targets(field, start, targets): #returns one-hot of action towards nearest target
    free_space=(field==0)
    if len(targets) == 0: return [0,0,0,0]

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            best = current
            break
        
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    
    current = best
    while True:
        if parent_dict[current] == start:
            xy=current
            break
        current = parent_dict[current]
    x=start[0]
    y=start[1]
    if xy == (x, y - 1):
        return [1,0,0,0]
    if xy == (x, y + 1):
        return [0,0,1,0]
    if xy == (x - 1, y):
        return [0,0,0,1]
    if xy == (x + 1, y):
        return [0,1,0,0]
    return [0,0,0,0]
