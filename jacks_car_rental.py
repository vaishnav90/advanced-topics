import numpy as np
import matplotlib.pyplot as plt

class JacksCarRental:
    def __init__(self, max_cars=20, max_move=5, theta=1e-6):
        self.max_cars = max_cars
        self.max_move = max_move
        self.theta = theta
        
        self.V = np.zeros((max_cars + 1, max_cars + 1))
        self.policy = np.zeros((max_cars + 1, max_cars + 1), dtype=int)
        
        self.lambda_rental_1 = 3
        self.lambda_rental_2 = 4
        self.lambda_return_1 = 3
        self.lambda_return_2 = 2
        
        self.reward_per_rental = 10
        self.cost_per_move = 2
        
    def poisson_probability(self, n, lambda_param):
        if n < 0:
            return 0
        return np.exp(-lambda_param) * (lambda_param ** n) / np.math.factorial(n)
    
    def get_rental_probabilities(self, lambda_param, max_rentals):
        probs = np.zeros(max_rentals + 1)
        for n in range(max_rentals + 1):
            probs[n] = self.poisson_probability(n, lambda_param)
        return probs
    
    def get_return_probabilities(self, lambda_param, max_returns):
        probs = np.zeros(max_returns + 1)
        for n in range(max_returns + 1):
            probs[n] = self.poisson_probability(n, lambda_param)
        return probs
    
    def calculate_expected_reward(self, state, action):
        cars_loc1, cars_loc2 = state
        
        cars_loc1 = max(0, min(self.max_cars, cars_loc1 - action))
        cars_loc2 = max(0, min(self.max_cars, cars_loc2 + action))
        
        reward = 0
        
        if action > 0:
            reward -= (action - 1) * self.cost_per_move
        elif action < 0:
            reward -= abs(action) * self.cost_per_move
        
        if cars_loc1 > 10:
            reward -= 4
        if cars_loc2 > 10:
            reward -= 4
        
        rental_probs_1 = self.get_rental_probabilities(self.lambda_rental_1, cars_loc1)
        rental_probs_2 = self.get_rental_probabilities(self.lambda_rental_2, cars_loc2)
        return_probs_1 = self.get_return_probabilities(self.lambda_return_1, self.max_cars)
        return_probs_2 = self.get_return_probabilities(self.lambda_return_2, self.max_cars)
        
        expected_reward = reward
        
        for rental1 in range(cars_loc1 + 1):
            for rental2 in range(cars_loc2 + 1):
                for return1 in range(self.max_cars + 1):
                    for return2 in range(self.max_cars + 1):
                        prob = (rental_probs_1[rental1] * rental_probs_2[rental2] * 
                               return_probs_1[return1] * return_probs_2[return2])
                        
                        rental_reward = (rental1 + rental2) * self.reward_per_rental
                        
                        next_cars1 = min(self.max_cars, cars_loc1 - rental1 + return1)
                        next_cars2 = min(self.max_cars, cars_loc2 - rental2 + return2)
                        
                        expected_reward += prob * (rental_reward + 0.9 * self.V[next_cars1, next_cars2])
        
        return expected_reward
    
    def policy_evaluation(self):
        while True:
            delta = 0
            for cars1 in range(self.max_cars + 1):
                for cars2 in range(self.max_cars + 1):
                    old_value = self.V[cars1, cars2]
                    action = self.policy[cars1, cars2]
                    self.V[cars1, cars2] = self.calculate_expected_reward((cars1, cars2), action)
                    delta = max(delta, abs(old_value - self.V[cars1, cars2]))
            
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        policy_stable = True
        for cars1 in range(self.max_cars + 1):
            for cars2 in range(self.max_cars + 1):
                old_action = self.policy[cars1, cars2]
                
                best_action = 0
                best_value = float('-inf')
                
                max_move_from_1 = min(cars1, self.max_move)
                max_move_from_2 = min(cars2, self.max_move)
                
                for action in range(-max_move_from_2, max_move_from_1 + 1):
                    if cars1 - action >= 0 and cars2 + action >= 0:
                        value = self.calculate_expected_reward((cars1, cars2), action)
                        if value > best_value:
                            best_value = value
                            best_action = action
                
                self.policy[cars1, cars2] = best_action
                
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable
    
    def policy_iteration(self):
        iteration = 0
        while True:
            iteration += 1
            
            self.policy_evaluation()
            
            if self.policy_improvement():
                break
    
    
    