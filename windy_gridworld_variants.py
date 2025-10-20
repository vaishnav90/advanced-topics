import numpy as np
import matplotlib.pyplot as plt

class WindyGridworld:
    def __init__(self, wind_strengths, stochastic=False):
        self.height = 7
        self.width = 10
        self.start = (3, 0)
        self.goal = (3, 7)
        self.wind_strengths = wind_strengths
        self.stochastic = stochastic
        self.actions_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.actions_9 = self.actions_8 + [(0, 0)]
        self.action_names_8 = ['Up-Left', 'Up', 'Up-Right', 'Left', 'Right', 'Down-Left', 'Down', 'Down-Right']
        self.action_names_9 = self.action_names_8 + ['Stay']
        
    def get_wind_effect(self, col):
        base_wind = self.wind_strengths[col]
        if not self.stochastic:
            return base_wind
        prob = np.random.random()
        if prob < 1/3:
            return base_wind - 1
        elif prob < 2/3:
            return base_wind
        else:
            return base_wind + 1
    
    def step(self, state, action, action_type='8'):
        row, col = state
        action_list = self.actions_8 if action_type == '8' else self.actions_9
        dr, dc = action_list[action]
        new_row = max(0, min(self.height - 1, row + dr))
        new_col = max(0, min(self.width - 1, col + dc))
        wind_effect = self.get_wind_effect(new_col)
        new_row = max(0, min(self.height - 1, new_row - wind_effect))
        done = (new_row, new_col) == self.goal
        reward = -1
        return (new_row, new_col), reward, done
    
    def reset(self):
        return self.start

class BaseAgent:
    def __init__(self, env, alpha=0.5, epsilon=0.1, action_type='8'):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.action_type = action_type
        self.num_actions = 8 if action_type == '8' else 9
        self.Q = {}
        self.episode_lengths = []
        self.returns = []
        
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
    
    def generate_trajectory(self, max_steps=100):
        state = self.env.reset()
        trajectory = [state]
        for _ in range(max_steps):
            best_action = 0
            best_value = float('-inf')
            for action in range(self.num_actions):
                q_value = self.get_q_value(state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            next_state, reward, done = self.env.step(state, best_action, self.action_type)
            trajectory.append(next_state)
            state = next_state
            if done:
                break
        return trajectory

class SARSAAgent(BaseAgent):
    def train(self, num_episodes=8000):
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.epsilon_greedy_action(state)
            episode_length = 0
            episode_return = 0
            while True:
                next_state, reward, done = self.env.step(state, action, self.action_type)
                episode_return += reward
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
            self.returns.append(episode_return)

class QLearningAgent(BaseAgent):
    def train(self, num_episodes=8000):
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_length = 0
            episode_return = 0
            while True:
                action = self.epsilon_greedy_action(state)
                next_state, reward, done = self.env.step(state, action, self.action_type)
                episode_return += reward
                if done:
                    q_value = self.get_q_value(state, action)
                    self.set_q_value(state, action, q_value + self.alpha * (reward - q_value))
                    break
                current_q = self.get_q_value(state, action)
                max_next_q = max([self.get_q_value(next_state, a) for a in range(self.num_actions)])
                new_q = current_q + self.alpha * (reward + max_next_q - current_q)
                self.set_q_value(state, action, new_q)
                state = next_state
                episode_length += 1
            self.episode_lengths.append(episode_length)
            self.returns.append(episode_return)

class ExpectedSARSAAgent(BaseAgent):
    def train(self, num_episodes=8000):
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_length = 0
            episode_return = 0
            while True:
                action = self.epsilon_greedy_action(state)
                next_state, reward, done = self.env.step(state, action, self.action_type)
                episode_return += reward
                if done:
                    q_value = self.get_q_value(state, action)
                    self.set_q_value(state, action, q_value + self.alpha * (reward - q_value))
                    break
                current_q = self.get_q_value(state, action)
                q_values = [self.get_q_value(next_state, a) for a in range(self.num_actions)]
                best_action = np.argmax(q_values)
                expected_q = (1 - self.epsilon) * q_values[best_action] + self.epsilon * np.mean(q_values)
                new_q = current_q + self.alpha * (reward + expected_q - current_q)
                self.set_q_value(state, action, new_q)
                state = next_state
                episode_length += 1
            self.episode_lengths.append(episode_length)
            self.returns.append(episode_return)

class DoubleQLearningAgent(BaseAgent):
    def __init__(self, env, alpha=0.5, epsilon=0.1, action_type='8'):
        super().__init__(env, alpha, epsilon, action_type)
        self.Q1 = {}
        self.Q2 = {}
    
    def get_q_value(self, state, action):
        return (self.Q1.get((state, action), 0.0) + self.Q2.get((state, action), 0.0)) / 2
    
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
            episode_length = 0
            episode_return = 0
            while True:
                action = self.epsilon_greedy_action(state)
                next_state, reward, done = self.env.step(state, action, self.action_type)
                episode_return += reward
                if np.random.random() < 0.5:
                    current_q1 = self.Q1.get((state, action), 0.0)
                    max_action_idx = np.argmax([self.Q1.get((next_state, a), 0.0) for a in range(self.num_actions)])
                    max_action_q2 = self.Q2.get((next_state, max_action_idx), 0.0)
                    new_q1 = current_q1 + self.alpha * (reward + max_action_q2 - current_q1)
                    self.Q1[(state, action)] = new_q1
                else:
                    current_q2 = self.Q2.get((state, action), 0.0)
                    max_action_idx = np.argmax([self.Q2.get((next_state, a), 0.0) for a in range(self.num_actions)])
                    max_action_q1 = self.Q1.get((next_state, max_action_idx), 0.0)
                    new_q2 = current_q2 + self.alpha * (reward + max_action_q1 - current_q2)
                    self.Q2[(state, action)] = new_q2
                if done:
                    break
                state = next_state
                episode_length += 1
            self.episode_lengths.append(episode_length)
            self.returns.append(episode_return)

class MonteCarloAgent(BaseAgent):
    def __init__(self, env, alpha=0.5, epsilon=0.1, action_type='8'):
        super().__init__(env, alpha, epsilon, action_type)
        self.returns_dict = {}
    
    def train(self, num_episodes=8000):
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_length = 0
            episode_return = 0
            max_steps = 1000
            step_count = 0
            while step_count < max_steps:
                action = self.epsilon_greedy_action(state)
                next_state, reward, done = self.env.step(state, action, self.action_type)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_return += reward
                episode_length += 1
                step_count += 1
                if done:
                    break
                state = next_state
            if len(episode_states) > 0:
                G = 0
                for t in reversed(range(len(episode_states))):
                    G = episode_rewards[t] + G
                    state_action = (episode_states[t], episode_actions[t])
                    if state_action not in [(episode_states[i], episode_actions[i]) for i in range(t)]:
                        if state_action not in self.returns_dict:
                            self.returns_dict[state_action] = []
                        self.returns_dict[state_action].append(G)
                        avg_return = np.mean(self.returns_dict[state_action])
                        self.set_q_value(episode_states[t], episode_actions[t], avg_return)
            self.episode_lengths.append(episode_length)
            self.returns.append(episode_return)

class OptimisticInitialValuesAgent(SARSAAgent):
    def __init__(self, env, alpha=0.5, epsilon=0.1, action_type='8', optimistic_value=5.0):
        super().__init__(env, alpha, epsilon, action_type)
        self.optimistic_value = optimistic_value
    
    def get_q_value(self, state, action):
        return self.Q.get((state, action), self.optimistic_value)

class DynamicProgrammingAgent(BaseAgent):
    def __init__(self, env, action_type='8'):
        super().__init__(env, 0.0, 0.0, action_type)
        self.policy = {}
        self.value_function = {}
    
    def train(self, num_iterations=100):
        for i in range(self.env.height):
            for j in range(self.env.width):
                self.value_function[(i, j)] = 0.0
        for iteration in range(num_iterations):
            new_values = {}
            for i in range(self.env.height):
                for j in range(self.env.width):
                    state = (i, j)
                    if state == self.env.goal:
                        new_values[state] = 0.0
                        continue
                    max_value = float('-inf')
                    best_action = 0
                    for action in range(self.num_actions):
                        next_state, reward, done = self.env.step(state, action, self.action_type)
                        value = reward + self.value_function.get(next_state, 0.0)
                        if value > max_value:
                            max_value = value
                            best_action = action
                    new_values[state] = max_value
                    self.policy[state] = best_action
            self.value_function = new_values
    
    def generate_trajectory(self, max_steps=100):
        state = self.env.reset()
        trajectory = [state]
        for _ in range(max_steps):
            action = self.policy.get(state, 0)
            next_state, reward, done = self.env.step(state, action, self.action_type)
            trajectory.append(next_state)
            state = next_state
            if done:
                break
        return trajectory

def compare_algorithms():
    wind_strengths = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    env = WindyGridworld(wind_strengths, stochastic=False)
    
    agents = {
        'SARSA': SARSAAgent(env, alpha=0.5, epsilon=0.1, action_type='9'),
        'Q-Learning': QLearningAgent(env, alpha=0.5, epsilon=0.1, action_type='9'),
        'Expected SARSA': ExpectedSARSAAgent(env, alpha=0.5, epsilon=0.1, action_type='9'),
        'Double Q-Learning': DoubleQLearningAgent(env, alpha=0.5, epsilon=0.1, action_type='9'),
        'Monte Carlo': MonteCarloAgent(env, alpha=0.5, epsilon=0.1, action_type='9'),
        'Optimistic IV': OptimisticInitialValuesAgent(env, alpha=0.5, epsilon=0.1, action_type='9', optimistic_value=5.0)
    }
    
    dp_agent = DynamicProgrammingAgent(env, action_type='9')
    print("Training Dynamic Programming...")
    dp_agent.train(num_iterations=100)
    dp_trajectory = dp_agent.generate_trajectory()
    agents['Dynamic Programming'] = dp_agent
    
    results = {}
    for name, agent in agents.items():
        if name != 'Dynamic Programming':
            print(f"Training {name}...")
            agent.train(num_episodes=8000)
        trajectory = agent.generate_trajectory()
        results[name] = {
            'trajectory': trajectory,
            'episode_lengths': agent.episode_lengths,
            'returns': agent.returns
        }
        print(f"{name}: {len(trajectory)-1} steps")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        if len(result['episode_lengths']) > 0:
            window_size = 100
            if len(result['episode_lengths']) >= window_size:
                smoothed = []
                for i in range(window_size, len(result['episode_lengths'])):
                    smoothed.append(np.mean(result['episode_lengths'][i-window_size:i]))
                plt.semilogx(range(window_size, len(result['episode_lengths'])), smoothed, label=name, linewidth=2)
            else:
                plt.semilogx(result['episode_lengths'], label=name, linewidth=2)
    plt.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Theoretical Minimum (15)')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Episode Length')
    plt.title('Learning Curves - Episode Lengths (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        if len(result['returns']) > 0:
            window_size = 100
            if len(result['returns']) >= window_size:
                smoothed = []
                for i in range(window_size, len(result['returns'])):
                    smoothed.append(np.mean(result['returns'][i-window_size:i]))
                plt.semilogx(range(window_size, len(result['returns'])), smoothed, label=name, linewidth=2)
            else:
                plt.semilogx(result['returns'], label=name, linewidth=2)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Average Return')
    plt.title('Learning Curves - Returns (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    final_performance = []
    names = []
    for name, result in results.items():
        if len(result['episode_lengths']) > 0:
            final_avg = np.mean(result['episode_lengths'][-min(1000, len(result['episode_lengths'])):])
            final_performance.append(final_avg)
            names.append(name)
    if final_performance:
        bars = plt.bar(names, final_performance)
        plt.ylabel('Final Average Episode Length')
        plt.title('Final Performance Comparison')
        plt.xticks(rotation=45)
        for bar, value in zip(bars, final_performance):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom')
    
    plt.subplot(2, 2, 4)
    optimal_paths = []
    names = []
    for name, result in results.items():
        optimal_paths.append(len(result['trajectory'])-1)
        names.append(name)
    bars = plt.bar(names, optimal_paths)
    plt.ylabel('Optimal Path Length')
    plt.title('Optimal Path Comparison')
    plt.xticks(rotation=45)
    for bar, value in zip(bars, optimal_paths):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    results = compare_algorithms()
