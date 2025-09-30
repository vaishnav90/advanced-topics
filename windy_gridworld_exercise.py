import numpy as np
import matplotlib.pyplot as plt

class WindyGridworld:
    def __init__(self, wind_strengths):
        self.height = 7
        self.width = 10
        self.start = (3, 0)
        self.goal = (3, 7)
        self.wind_strengths = wind_strengths
        self.actions_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.actions_9 = self.actions_8 + [(0, 0)]
        self.action_names_8 = ['Up-Left', 'Up', 'Up-Right', 'Left', 'Right', 'Down-Left', 'Down', 'Down-Right']
        self.action_names_9 = self.action_names_8 + ['Stay']
    
    def step(self, state, action, action_type='8'):
        row, col = state
        action_list = self.actions_8 if action_type == '8' else self.actions_9
        dr, dc = action_list[action]
        new_row = max(0, min(self.height - 1, row + dr))
        new_col = max(0, min(self.width - 1, col + dc))
        wind_effect = self.wind_strengths[new_col]
        new_row = max(0, min(self.height - 1, new_row - wind_effect))
        done = (new_row, new_col) == self.goal
        reward = -1
        return (new_row, new_col), reward, done
    
    def reset(self):
        return self.start

class SARSA:
    def __init__(self, env, alpha=0.5, epsilon=0.1, action_type='8'):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.action_type = action_type
        self.num_actions = 8 if action_type == '8' else 9
        self.Q = {}
        self.episode_lengths = []
    
    def get_q_value(self, state, action):
        return self.Q.get((state, action), 0.0)
    
    def set_q_value(self, state, action, value):
        self.Q[(state, action)] = value
    
    def epsilon_greedy_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        best_action = 0
        best_value = float('-inf')
        for action in range(self.num_actions):
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action
    
    def train(self, num_episodes=8000):
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.epsilon_greedy_action(state)
            episode_length = 0
            while True:
                next_state, reward, done = self.env.step(state, action, self.action_type)
                if done:
                    q_value = self.get_q_value(state, action)
                    self.set_q_value(state, action, q_value + self.alpha * (reward - q_value))
                    break
                next_action = self.epsilon_greedy_action(next_state)
                current_q = self.get_q_value(state, action)
                next_q = self.get_q_value(next_state, next_action)
                new_q = current_q + self.alpha * (reward + next_q - current_q)
                self.set_q_value(state, action, new_q)
                state = next_state
                action = next_action
                episode_length += 1
            self.episode_lengths.append(episode_length)
    
    def get_optimal_policy(self):
        policy = {}
        for i in range(self.env.height):
            for j in range(self.env.width):
                state = (i, j)
                best_action = 0
                best_value = float('-inf')
                for action in range(self.num_actions):
                    q_value = self.get_q_value(state, action)
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                policy[state] = best_action
        return policy
    
    def generate_trajectory(self, max_steps=100):
        state = self.env.reset()
        trajectory = [state]
        for _ in range(max_steps):
            policy = self.get_optimal_policy()
            action = policy[state]
            next_state, reward, done = self.env.step(state, action, self.action_type)
            trajectory.append(next_state)
            state = next_state
            if done:
                break
        return trajectory

def print_optimal_path(trajectory, action_type):
    action_names = env_8.action_names_8 if action_type == '8' else env_9.action_names_9
    actions = env_8.actions_8 if action_type == '8' else env_9.actions_9
    path_actions = []
    for i in range(len(trajectory) - 1):
        current = trajectory[i]
        next_pos = trajectory[i + 1]
        print(current, next_pos)
        for action_idx, (dr, dc) in enumerate(actions):
            test_row = max(0, min(6, current[0] + dr))
            test_col = max(0, min(9, current[1] + dc))
            wind_effect = wind_strengths[test_col]
            final_row = max(0, min(6, test_row - wind_effect))
            if (final_row, test_col) == next_pos:
                path_actions.append(action_names[action_idx])
                break
    return path_actions

wind_strengths = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
env_8 = WindyGridworld(wind_strengths)
env_9 = WindyGridworld(wind_strengths)

sarsa_8 = SARSA(env_8, alpha=0.5, epsilon=0.1, action_type='8')
sarsa_8.train(num_episodes=8000)

sarsa_9 = SARSA(env_9, alpha=0.5, epsilon=0.1, action_type='9')
sarsa_9.train(num_episodes=8000)

trajectory_8 = sarsa_8.generate_trajectory()
trajectory_9 = sarsa_9.generate_trajectory()

path_8 = print_optimal_path(trajectory_8, '8')
path_9 = print_optimal_path(trajectory_9, '9')

print(f"8 Actions: {len(trajectory_8)-1} steps")
print(f"Action sequence: {' -> '.join(path_8)}")
print(f"\n9 Actions: {len(trajectory_9)-1} steps") 
print(f"Action sequence: {' -> '.join(path_9)}")

plt.figure(figsize=(12, 8))
window_size = 100
smoothed_8 = []
smoothed_9 = []
for i in range(window_size, len(sarsa_8.episode_lengths)):
    smoothed_8.append(np.mean(sarsa_8.episode_lengths[i-window_size:i]))
    smoothed_9.append(np.mean(sarsa_9.episode_lengths[i-window_size:i]))

plt.plot(range(window_size, len(sarsa_8.episode_lengths)), smoothed_8, label='8 Actions', linewidth=2)
plt.plot(range(window_size, len(sarsa_9.episode_lengths)), smoothed_9, label='9 Actions', linewidth=2)
plt.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Minimum (15 steps)')
plt.xlabel('Episodes')
plt.ylabel('Episode Length (Time Steps)')
plt.title('Windy Gridworld: 8 vs 9 Actions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
