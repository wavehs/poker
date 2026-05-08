import time
import sys
import random

valid_h_hands = [(random.randint(0, 51), random.randint(0, 51)) for _ in range(1000)]
v_hand = (1, 2)

def set_way(sims):
    count = 0
    for _ in range(sims):
        sampled_board = [3, 4, 5, 6, 7]
        forbidden = {v_hand[0], v_hand[1], *sampled_board}
        for h_hand in valid_h_hands:
            if h_hand[0] in forbidden or h_hand[1] in forbidden:
                continue
            count += 1
    return count

def array_way(sims):
    count = 0
    for _ in range(sims):
        sampled_board = [3, 4, 5, 6, 7]
        forbidden = [False] * 52
        forbidden[v_hand[0]] = True
        forbidden[v_hand[1]] = True
        for s in sampled_board:
            forbidden[s] = True
        for h_hand in valid_h_hands:
            if forbidden[h_hand[0]] or forbidden[h_hand[1]]:
                continue
            count += 1
    return count

t0 = time.time()
set_way(10000)
print("Set way:", time.time() - t0)

t0 = time.time()
array_way(10000)
print("Array way:", time.time() - t0)
