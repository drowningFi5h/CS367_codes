from collections import deque

def make_initial(n):
  state = "e"*n + "_" + "w"*n  
  return state

def emptyStone(state):
  for i in len(state):
    if state[i]=="_":
      return i

def get_neighbors(state):
  neighbors = []
  L = len(state)
  state_list = list(state) 

  for i in range(L):
    if state[i] == "e":
      if i + 1 < L and state[i+1] == "_":
        new_state = state_list[:]
        new_state[i], new_state[i+1] = new_state[i+1], new_state[i]
        neighbors.append(("E from {} to {}".format(i, i+1), "".join(new_state)))

      if i + 2 < L and state[i+2] == "_" and state[i+1] != "_":
        new_state = state_list[:]
        new_state[i], new_state[i+2] = new_state[i+2], new_state[i]
        neighbors.append(("E from {} to {}".format(i, i+2), "".join(new_state)))

    elif state[i] == "w":
      
      if i - 1 >= 0 and state[i-1] == "_":
        new_state = state_list[:]
        new_state[i], new_state[i-1] = new_state[i-1], new_state[i]
        neighbors.append(("W from {} to {}".format(i, i-1), "".join(new_state)))

      if i - 2 >= 0 and state[i-2] == "_" and state[i-1] != "_":
        new_state = state_list[:]
        new_state[i], new_state[i-2] = new_state[i-2], new_state[i]
        neighbors.append(("W from {} to {}".format(i, i-2), "".join(new_state)))

  return neighbors

def bfs(start , goal):
  queue = deque([start])
  visited = {start : (None,None)}

  while queue:
    state = queue.popleft()

    if state == goal:
      path = []
      moves = []
      while visited[state][0] is not None:
        parent , move = visited[state]
        path.append(state)
        moves.append(move)
        state = parent
      path.append(start)
      path.reverse()
      moves.reverse()
      return path , moves
      
    
    for move,new_state in get_neighbors(state):
      if new_state not in visited:
        visited[new_state] = (state,move)
        queue.append(new_state)
  return None,None


n = 3
start = make_initial(n)
goal = "w"*n + "_" + "e"*n

path, moves = bfs(start, goal)

print("Path of states:")
for p in path:
    print(p)

print("\nMoves:")
for m in moves:
    print(m)
