import events as e
import numpy as np

#our reward functions, called only at the end of a game to keep calculations during playing minimal
#return all rewards of a single game

def advancedrewards(allevents):   
    rewards=np.zeros(len(allevents))
    for i in range(len(allevents)):
        events=allevents[i]
        if e.COIN_COLLECTED in events:
            rewards[i]+=3
        if e.CRATE_DESTROYED in events:
            rewards[i-4]+=1
        if e.INVALID_ACTION in events:
            rewards[i]+=-5
        if e.KILLED_OPPONENT in events:
            rewards[i-4]+=10
            rewards[i-3]+=3
        if e.KILLED_SELF in events:
            rewards[i-4]+=-400
            rewards[i-3]+=-200
            rewards[i-2]+=-200
            rewards[i-1]+=-200
            rewards[i]+=-200
        if e.GOT_KILLED in events and not e.KILLED_SELF in events:
            rewards[i-3]+=-200
            rewards[i-2]+=-300
            rewards[i-1]+=-300
            rewards[i]+=-400
    return rewards

def immediaterewards(allevents):
    rewards=np.zeros(len(allevents))
    for i in range(len(allevents)):
        events=allevents[i]
        if e.BOMB_DROPPED in events:
            rewards[i]+=1
        if e.COIN_COLLECTED in events:
            rewards[i]+=200
        if e.CRATE_DESTROYED in events:
            rewards[i]+=50
        if e.INVALID_ACTION in events:
            rewards[i]+=-5
        if e.KILLED_OPPONENT in events:
            rewards[i]+=700
        if e.KILLED_SELF in events:
            rewards[i]-=1000
        elif e.GOT_KILLED in events:
            rewards[i]-=500
    return rewards
