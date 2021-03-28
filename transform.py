import numpy as np
from random import shuffle

#all the functions to compute features from states

def nostone(game_state):  #feature 6.3
    field=game_state["field"]
    me=game_state["self"][3]
    bombs=game_state["bombs"]
    others=game_state["others"]
    if len(game_state["coins"])>0:
        field[tuple(zip(*game_state["coins"]))]=10
    for other in others:
        field[other[3]]=2
    for bomb in bombs:
        field[bomb[0]]=-5+bomb[1]
    field[me]+=100
    flat=field.flatten()
    
    return flat[flat!=-1].reshape(1,-1)

def subfield(game_state,dist,nearest=False): #feature 6.1 iff nearest==False and 6.2 iff nearest==True, dist is visual range
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
    return np.concatenate((field,nearest),axis=None).reshape(1,-1)

def minimal(game_state): #feature 6.4
    field=game_state["field"]
    x,y=game_state["self"][3]
    coins=game_state["coins"]
    crates=[]
    for i in range(17):
        for j in range(17):
            if field[i,j]==1:
                crates.append((i,j))
    
    bombfield=np.zeros((17,17))
    for bomb in game_state["bombs"]:
        bx,by=bomb[0]
        t=bomb[1]
        bombfield[bx,by]=t+15
        field[bx,by]=-2
        if field[bx+1,by]!=-1:
            bombfield[bx+1,by]=max(t+10,bombfield[bx+1,by])
            if field[bx+2,by]!=-1:
                bombfield[bx+2,by]=max(t+5,bombfield[bx+2,by])
                if field[bx+3,by]!=-1:
                    bombfield[bx+3,by]=max(t,bombfield[bx+3,by])
        if field[bx-1,by]!=-1:
            bombfield[bx-1,by]=max(t+10,bombfield[bx-1,by])
            if field[bx-2,by]!=-1:
                bombfield[bx-2,by]=max(t+5,bombfield[bx-2,by])
                if field[bx-3,by]!=-1:
                    bombfield[bx-3,by]=max(t,bombfield[bx-3,by])
        if field[bx,by+1]!=-1:
            bombfield[bx,by+1]=max(t+10,bombfield[bx,by+1])
            if field[bx,by+2]!=-1:
                bombfield[bx,by+2]=max(t+5,bombfield[bx,by+2])
                if field[bx,by+3]!=-1:
                    bombfield[bx,by+3]=max(t,bombfield[bx,by+3])
        if field[bx,by-1]!=-1:
            bombfield[bx,by-1]=max(t+10,bombfield[bx,by-1])
            if field[bx,by-2]!=-1:
                bombfield[bx,by-2]=max(t+50,bombfield[bx,by-2])
                if field[bx,by-3]!=-1:
                    bombfield[bx,by-3]=max(t+10,bombfield[bx,by-3])

    others=[]
    for other in game_state["others"]:
        field[other[3]]=-2
        others.append(other[3])

    possibles=np.zeros(5)

    if field[x,y+1]==0:
        possibles[2]=1
    if field[x,y-1]==0:
        possibles[0]=1
    if field[x+1,y]==0:
        possibles[1]=1
    if field[x-1,y]==0:
        possibles[3]=1
    if game_state["self"][2]:
        possibles[4]=1

    crf=look_for_targets(field,(x,y),crates)
    of=look_for_targets(field,(x,y),others)
    cof=look_for_targets(field,(x,y),coins)
    bf=np.zeros(5)
    
    bf[4]=bombfield[x,y]
    bf[0]=bombfield[x+1,y]
    bf[1]=bombfield[x-1,y]
    bf[2]=bombfield[x,y+1]
    bf[3]=bombfield[x,y-1]
        
    feature=np.concatenate((possibles,bf,crf,cof,of))
    #print(feature)
    return feature.reshape(1,-1)

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
