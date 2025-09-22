import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Blackjack:
    def __init__(self):
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self.Q = {}
        self.pi = {}
        self.returns = {}
        self.eps = 1000000
    def get_card_value(self, card):
        return 1 if card == 1 else card
    def get_hand_value(self, hand):
        total = sum(hand)
        aces = hand.count(1)
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total, aces > 0
    def get_state(self, p_sum, d_card, u_ace):
        return (p_sum, d_card, u_ace)
    def get_action(self, state, eps=0.1):
        if np.random.random() < eps:
            return np.random.choice([0, 1])
        else:
            if state in self.pi:
                return self.pi[state]
            else:
                return 0 if state[0] >= 20 else 1
    def dealer_play(self, d_hand):
        while True:
            total, _ = self.get_hand_value(d_hand)
            if total >= 17:
                break
            d_hand.append(np.random.choice(self.deck))
        return d_hand
    def play_episode(self, es=True):
        if es:
            p_sum = np.random.randint(11, 22)
            d_card = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            u_ace = np.random.choice([True, False])
        else:
            p_sum = 11
            d_card = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            u_ace = False
        state = self.get_state(p_sum, d_card, u_ace)
        action = self.get_action(state, eps=0.1)
        p_hand = []
        if u_ace:
            if p_sum == 11:
                p_hand = [1]
            else:
                p_hand = [1, p_sum - 11]
        else:
            if p_sum == 11:
                p_hand = [11]
            else:
                p_hand = [p_sum]
        if action == 1:
            p_hand.append(np.random.choice(self.deck))
        p_total, p_u_ace = self.get_hand_value(p_hand)
        if p_total > 21:
            return [(state, action, -1)]
        d_hand = [d_card, np.random.choice(self.deck)]
        d_hand = self.dealer_play(d_hand)
        d_total, _ = self.get_hand_value(d_hand)
        if d_total > 21:
            reward = 1
        elif p_total > d_total:
            reward = 1
        elif p_total < d_total:
            reward = -1
        else:
            reward = 0
        return [(state, action, reward)]
    def monte_carlo_es(self):
        for ep in range(self.eps):
            ep_data = self.play_episode(es=True)
            for state, action, reward in ep_data:
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(reward)
                self.Q[(state, action)] = np.mean(self.returns[(state, action)])
                if state not in self.pi:
                    self.pi[state] = 0
                if self.Q.get((state, 0), -float('inf')) > self.Q.get((state, 1), -float('inf')):
                    self.pi[state] = 0
                else:
                    self.pi[state] = 1
    def get_value_function(self):
        V = {}
        for state in self.pi:
            V[state] = max(self.Q.get((state, 0), 0), self.Q.get((state, 1), 0))
        return V
    def plot_figure_5_2(self):
        V = self.get_value_function()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        p_sums = list(range(11, 22))
        d_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pi_u = np.zeros((len(p_sums), len(d_cards)))
        val_u = np.zeros((len(p_sums), len(d_cards)))
        pi_nu = np.zeros((len(p_sums), len(d_cards)))
        val_nu = np.zeros((len(p_sums), len(d_cards)))
        for i, ps in enumerate(p_sums):
            for j, dc in enumerate(d_cards):
                state_u = (ps, dc, True)
                state_nu = (ps, dc, False)
                pi_u[i, j] = self.pi.get(state_u, 0)
                val_u[i, j] = V.get(state_u, 0)
                pi_nu[i, j] = self.pi.get(state_nu, 0)
                val_nu[i, j] = V.get(state_nu, 0)
        for i, ps in enumerate(p_sums):
            for j, dc in enumerate(d_cards):
                if pi_u[i, j] == 0:
                    axes[0, 0].plot([j-0.5, j+0.5], [i-0.5, i-0.5], 'k-', linewidth=2)
                    axes[0, 0].plot([j-0.5, j+0.5], [i+0.5, i+0.5], 'k-', linewidth=2)
                    axes[0, 0].plot([j-0.5, j-0.5], [i-0.5, i+0.5], 'k-', linewidth=2)
                    axes[0, 0].plot([j+0.5, j+0.5], [i-0.5, i+0.5], 'k-', linewidth=2)
        axes[0, 0].set_title('Usable ace')
        axes[0, 0].set_xlabel('Dealer showing')
        axes[0, 0].set_ylabel('Player sum')
        axes[0, 0].set_xticks(range(len(d_cards)))
        axes[0, 0].set_xticklabels(['A'] + [str(i) for i in d_cards[1:]])
        axes[0, 0].set_yticks(range(len(p_sums)))
        axes[0, 0].set_yticklabels(p_sums)
        axes[0, 0].set_xlim(-0.5, len(d_cards)-0.5)
        axes[0, 0].set_ylim(-0.5, len(p_sums)-0.5)
        axes[0, 0].invert_yaxis()
        X, Y = np.meshgrid(d_cards, p_sums)
        ax1 = fig.add_subplot(2, 2, 2, projection='3d')
        ax1.plot_surface(X, Y, val_u, cmap='viridis', alpha=0.8)
        ax1.set_title('Usable ace')
        ax1.set_xlabel('Dealer showing')
        ax1.set_ylabel('Player sum')
        ax1.set_zlabel('Value')
        for i, ps in enumerate(p_sums):
            for j, dc in enumerate(d_cards):
                if pi_nu[i, j] == 0:
                    axes[1, 0].plot([j-0.5, j+0.5], [i-0.5, i-0.5], 'k-', linewidth=2)
                    axes[1, 0].plot([j-0.5, j+0.5], [i+0.5, i+0.5], 'k-', linewidth=2)
                    axes[1, 0].plot([j-0.5, j-0.5], [i-0.5, i+0.5], 'k-', linewidth=2)
                    axes[1, 0].plot([j+0.5, j+0.5], [i-0.5, i+0.5], 'k-', linewidth=2)
        axes[1, 0].set_title('No usable ace')
        axes[1, 0].set_xlabel('Dealer showing')
        axes[1, 0].set_ylabel('Player sum')
        axes[1, 0].set_xticks(range(len(d_cards)))
        axes[1, 0].set_xticklabels(['A'] + [str(i) for i in d_cards[1:]])
        axes[1, 0].set_yticks(range(len(p_sums)))
        axes[1, 0].set_yticklabels(p_sums)
        axes[1, 0].set_xlim(-0.5, len(d_cards)-0.5)
        axes[1, 0].set_ylim(-0.5, len(p_sums)-0.5)
        axes[1, 0].invert_yaxis()
        ax2 = fig.add_subplot(2, 2, 4, projection='3d')
        ax2.plot_surface(X, Y, val_nu, cmap='viridis', alpha=0.8)
        ax2.set_title('No usable ace')
        ax2.set_xlabel('Dealer showing')
        ax2.set_ylabel('Player sum')
        ax2.set_zlabel('Value')
        plt.tight_layout()
        plt.show()

class CasinoBlackjack(Blackjack):
    def __init__(self):
        super().__init__()
        self.eps = 1000000
    def play_episode_casino(self, es=True):
        if es:
            p_sum = np.random.randint(11, 22)
            d_card = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            u_ace = np.random.choice([True, False])
        else:
            p_sum = 11
            d_card = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            u_ace = False
        state = self.get_state(p_sum, d_card, u_ace)
        action = self.get_action(state, eps=0.1)
        p_hand = []
        if u_ace:
            if p_sum == 11:
                p_hand = [1]
            else:
                p_hand = [1, p_sum - 11]
        else:
            if p_sum == 11:
                p_hand = [11]
            else:
                p_hand = [p_sum]
        
        if action == 1:
            p_hand.append(np.random.choice(self.deck))
        p_total, p_u_ace = self.get_hand_value(p_hand)
        if p_total > 21:
            return [(state, action, -1)]
        if p_total == 21:
            return [(state, action, 1.5)]
        d_hand = [d_card, np.random.choice(self.deck)]
        d_hand = self.dealer_play(d_hand)
        d_total, _ = self.get_hand_value(d_hand)
        if d_total > 21:
            reward = 1
        elif p_total > d_total:
            reward = 1
        elif p_total < d_total:
            reward = -1
        else:
            reward = -1
        return [(state, action, reward)]
    def monte_carlo_es_casino(self):
        for ep in range(self.eps):
            ep_data = self.play_episode_casino(es=True)
            for state, action, reward in ep_data:
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(reward)
                self.Q[(state, action)] = np.mean(self.returns[(state, action)])
                if state not in self.pi:
                    self.pi[state] = 0
                if self.Q.get((state, 0), -float('inf')) > self.Q.get((state, 1), -float('inf')):
                    self.pi[state] = 0
                else:
                    self.pi[state] = 1
    
    def calculate_expected_value(self, n_sims=100000):
        total_reward = 0
        for _ in range(n_sims):
            p_sum = np.random.randint(11, 22)
            d_card = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            u_ace = np.random.choice([True, False])
            state = self.get_state(p_sum, d_card, u_ace)
            action = self.pi.get(state, 0)
            p_hand = []
            if u_ace:
                if p_sum == 11:
                    p_hand = [1]
                else:
                    p_hand = [1, p_sum - 11]
            else:
                if p_sum == 11:
                    p_hand = [11]
                else:
                    p_hand = [p_sum]
            if action == 1:
                p_hand.append(np.random.choice(self.deck))
            p_total, _ = self.get_hand_value(p_hand)
            if p_total > 21:
                total_reward -= 1
                continue
            if p_total == 21:
                total_reward += 1.5
                continue
            d_hand = [d_card, np.random.choice(self.deck)]
            d_hand = self.dealer_play(d_hand)
            d_total, _ = self.get_hand_value(d_hand)
            if d_total > 21:
                total_reward += 1
            elif p_total > d_total:
                total_reward += 1
            elif p_total < d_total:
                total_reward -= 1
            else:
                total_reward -= 1
        return total_reward / n_sims
if __name__ == "__main__":
    blackjack = Blackjack()
    blackjack.monte_carlo_es()
    blackjack.plot_figure_5_2()
    casino_blackjack = CasinoBlackjack()
    casino_blackjack.monte_carlo_es_casino()
    expected_value = casino_blackjack.calculate_expected_value()
    print(f"optimal play: {expected_value:.4f}")
