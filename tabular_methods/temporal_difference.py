import numpy as np
from tqdm.notebook import tqdm

# This class define the Temporal-Difference agent

class Qlearning_agent(object):

    def solve(self, env, total_episodes=1000, alpha=0.9):
        """
        Solve a given Maze environment using Temporal Difference learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        """
        E = 0.3
        Q = np.random.rand(env.get_state_size(), env.get_action_size())
        
        p_behaviour = np.random.rand(env.get_state_size(), env.get_action_size())
        p_behaviour /= np.sum(p_behaviour, axis=1)[:,None] 
        p_target = np.random.rand(env.get_state_size(), env.get_action_size())
        p_target /= np.sum(p_target, axis=1)[:,None] 
        
        values = []
        total_rewards = []
        
        ####
        # Off-policy TD control: Q-learning
        ####
        n_episodes = 0
        
        for n_episodes in tqdm(range(total_episodes), unit="episode", leave=False):
            total_reward = 0.0
            
            t, state, reward, done = env.reset()
            
            while not done:
                a_behaviour = np.random.choice(list(range(4)), p=p_behaviour[state])
                
                t, next_state, reward, done = env.step(a_behaviour)
                total_reward += reward
                
                a_target = np.argmax(Q[next_state,:])
                p_target[next_state] *= 0.0
                p_target[next_state][a_target] = 1.0
                
                Q[state, a_behaviour] = Q[state, a_behaviour] + alpha * (reward + env.get_gamma()*Q[next_state, a_target] - Q[state, a_behaviour])
                
                # Update e-greedy policy
                a_opt = np.argmax(Q[state,:])
                for a in range(env.get_action_size()):
                    if a == a_opt:
                        p_behaviour[state][a] = 1 - E + E/env.get_action_size()
                    else:
                        p_behaviour[state][a] = E/env.get_action_size()

                state = next_state
            
            policy = p_target
            values.append(np.sum(policy*Q, axis=1))
            total_rewards.append(total_reward)
            n_episodes += 1
            
        
        return policy, values, total_rewards

class SARSA_agent(object):

    def solve(self, env, total_episodes=1000, alpha=0.9):
        """
        Solve a given Maze environment using Temporal Difference learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        """
        E = 0.1
        Q = np.random.rand(env.get_state_size(), env.get_action_size())
        
        policy = np.random.rand(env.get_state_size(), env.get_action_size())
        policy /= np.sum(policy, axis=1)[:,None] 
        policy = np.ones((env.get_state_size(), env.get_action_size()))*0.25
        
        values = []
        total_rewards = []
        
        ####
        # On-policy TD control: SARSA
        #
        ####
        for n_episodes in tqdm(range(total_episodes), unit="episode", leave=False):
            
            total_reward = 0.0
            
            t, state, reward, done = env.reset()
            action = np.random.choice(list(range(4)), p=policy[state])

            if type(alpha) == str and alpha == 'Episode Inverse Decay':
                alpha_t = 1/(n_episodes+1)
            
            while not done:
                t, next_state, reward, done = env.step(action)
                total_reward += reward

                next_action = np.random.choice(list(range(4)), p=policy[next_state])
                Q[state, action] = Q[state, action] + alpha_t * (reward + env.get_gamma()*Q[next_state, next_action] - Q[state, action])
                
                # Update e-greedy policy
                a_opt = np.argmax(Q[state,:])
                for a in range(env.get_action_size()):
                    if a == a_opt:
                        policy[state][a] = 1 - E + E/env.get_action_size()
                    else:
                        policy[state][a] = E/env.get_action_size()

                state, action = next_state, next_action
            
            values.append(np.sum(policy*Q, axis=1))
            total_rewards.append(total_reward)
            n_episodes += 1
            
        
        return policy, values, total_rewards