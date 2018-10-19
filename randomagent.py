import gym
import gym.spaces
import numpy as np
import random


def randomagent(prints=False):
    # Initializing the list of scores
    scores = []
    
    # Creating the gym environment
    env = gym.make("Taxi-v2")

    # Amount of games the agent plays
    episodes = 50000

    # Maximum steps the agent has per episode
    max_steps = 100

    for episode in range(episodes):
        # Reset the state, done and score before every episode
        env.reset()
        done = False
        score = 0

        for _ in range(max_steps):
            # Act randomly until done or maximum steps reached
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        
        scores.append(score)
        if prints:
            print("Episode: {}/{}, score: {}".format(episode+1, episodes, score))

    return scores


if __name__ == "__main__":
    randomagent(prints=True)
