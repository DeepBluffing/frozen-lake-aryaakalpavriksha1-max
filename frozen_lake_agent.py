import gymnasium as gym
import numpy as np

def train_agent():
    # Initialize environment (is_slippery=False makes it deterministic for easier learning)
    env = gym.make("FrozenLake-v1", is_slippery=False)
    
    # Hyperparameters (TRY TWEAKING THEM!!!)
    num_episodes = 10000
    alpha = 0.8       # Learning rate
    gamma = 0.95      # Discount factor
    epsilon = 1.0     # Exploration rate
    min_epsilon = 0.01
    epsilon_decay = 0.995

    # Initialize Q-Table with zeros (16 states, 4 actions)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # ==========================================
            # TODO 1: Implement Epsilon-Greedy Action Selection
            # Hint: Generate a random number using np.random.uniform(0, 1). 
            # If it's less than epsilon, choose a random action (env.action_space.sample()).
            # Otherwise, choose the action with the highest Q-value for the current state (np.argmax).
            # ==========================================
           if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state, :])  # Exploit
            
            
            # Take the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ==========================================
            # TODO 2: The Bellman Equation Update
            # Update q_table[state, action] using alpha, gamma, the reward, and the max Q-value of the next_state.
            # ==========================================
            best_next_action = np.max(q_table[next_state, :])
            td_target = reward + gamma * best_next_action
            
            # Update the Q-Table using the TD Error
            q_table[state, action] += alpha * (td_target - q_table[state, action])
            
            state = next_state
            
            
            
        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table

if __name__ == "__main__":
    trained_q_table = train_agent()
    print("\nTraining complete! Final Q-Table:")
    print(trained_q_table)
