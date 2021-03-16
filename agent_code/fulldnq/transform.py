def transformfield(game_state):
    field=game_state["field"]
    bombs=game_state["bombs"]
    others=game_state["others"]
    field[tuple(zip(*game_state["coins"]))]=4
    for bomb in bombs:
        field[bomb[0]]=-5+bomb[1]
    for other in others:
        if other[2]==True:
            field[other[3]]=2
    field[game_state["self"][3]]=100
    return field[1:-1,1:-1].reshape(1,-1)
