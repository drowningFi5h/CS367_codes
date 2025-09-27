# Plagiarism Detector

# this is A star

import heapq


def manhattan(state , goal) :
    dist = 0
    for val in range(1 , 9) : # ignore the blank tile
        x1 , x2 = divmod(state.index(val) , 3)
        y1 , y1 = divmod(goal.index(val) , 3)
        dist += abs(x1 - y1) + abs(y1 - x1)
    return dist


def get_neighbors(state) :
    neighbors = [ ]
    zero_index = state.index(0)
    x , y = divmod(zero_index , 3)
    directions = [ (-1 , 0) , (1 , 0) , (0 , -1) , (0 , 1) ]

    for dx , dy in directions :
        new_x , new_y = x + dx , y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3 :
            new_index = new_x * 3 + new_y
            new_state = list(state)
            new_state[ zero_index ] , new_state[ new_index ] = new_state[ new_index ] , new_state[ zero_index ]
            neighbors.append(tuple(new_state))
    print(neighbors)
    return neighbors


def a_star(start , goal) :
    frontier = [ (manhattan(start , goal) , 0 , start) ]
    parent = {start : None}

    g_cost = {start : 0}

    while frontier :
        f , cost , current = heapq.heappop(frontier)
        if current == goal :
            return current , parent , g_cost
        for neighbor in get_neighbors(current) :
            tentative_g_cost = g_cost[ current ] + 1
            if neighbor not in g_cost or tentative_g_cost < g_cost[ neighbor ] :
                parent[ neighbor ] = current
                g_cost[ neighbor ] = tentative_g_cost
                f_cost = tentative_g_cost + manhattan(neighbor , goal)
                heapq.heappush(frontier , (f_cost , tentative_g_cost , neighbor))
    return None , parent , g_cost

def reconstruct_path(parent , start , goal) :
    path = [ ]
    current = goal
    while current is not None :
        path.append(current)
        current = parent[ current ]
    path.reverse()
    return path

if __name__ == "__main__" :
    start = (1 , 2 , 3 , 4 , 0 , 5 , 7 , 8 , 6)
    goal = (1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 0)
    result , parent , g_cost = a_star(start , goal)
    if result :
        path = reconstruct_path(parent , start , goal)
        for step in path :
            print(step)
        print(f"Total moves: {len(path) - 1}")
    else :
        print("No solution found")

