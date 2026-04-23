import random
import time


def simulate_with_shuffle():
    deck = list(range(45))
    sims = 100000
    cards_per_sim = 7
    t0 = time.perf_counter()
    for _ in range(sims):
        random.shuffle(deck)
        x = deck[:cards_per_sim]
    return time.perf_counter() - t0

def simulate_with_sample():
    deck = list(range(45))
    sims = 100000
    cards_per_sim = 7
    t0 = time.perf_counter()
    for _ in range(sims):
        x = random.sample(deck, cards_per_sim)
    return time.perf_counter() - t0

print("shuffle:", simulate_with_shuffle())
print("sample:", simulate_with_sample())
