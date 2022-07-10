import numpy as np
from tqdm.notebook import tqdm

# This class define the Monte-Carlo agent

class MC_agent(object):
    
    def generate_episode(self, env, policy):
        episode = []
        
        t, state, reward, done = env.reset()
        action = np.random.choice(list(range(4)), p=policy[state])
        episode.append((reward, state, action))
        total_reward = reward
        
        while not done:
            t, state, reward, done = env.step(action)
            action = np.random.choice(list(range(4)), p=policy[state])
            episode.append((reward, state, action))
            total_reward += reward
            
        return episode, total_reward
        
    def solve(self, env, total_episodes=1000, epsilon='1/cbrt(n_episodes)'):
        """
        Solve a given Maze environment using Monte Carlo learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        """
        # Initialisation
        Q = np.random.rand(env.get_state_size(), env.get_action_size())
        V = np.zeros(env.get_state_size())
        
        # Generate a random soft policy
        policy = np.random.rand(env.get_state_size(), env.get_action_size())
        policy /= np.sum(policy, axis=1)[:,None] 
        policy = np.ones((env.get_state_size(), env.get_action_size()))*0.25
        
        values = []
        total_rewards = []
        
        n_episodes = 0
        counts = dict()
        
        for n_episodes in tqdm(range(total_episodes), unit="episode", leave=False):
            
            episode, total_reward = self.generate_episode(env, policy)
            total_rewards.append(total_reward)
            
            G = 0
            if type(epsilon) == str and epsilon=='1/cbrt(n_episodes)':
                E = 1/np.cbrt(n_episodes+1)
            elif type(epsilon) == str and epsilon=='1/sqrt(n_episodes)':
                E = 1/np.sqrt(n_episodes+1)
            elif type(epsilon) == str and epsilon=='1/n_episodes':
                E = 1/(n_episodes+1)
            elif type(epsilon) == float:
                E = epsilon
            
            for t in range(len(episode)-2, -1, -1):
                G = env.get_gamma() * G + episode[t+1][0] 
                # State is saved in the second position of the tuple storing the trace, 
                # and action in the third one
                s_t = episode[t][1] 
                a_t = episode[t][2]
                
                # Condition to accomplish on-policy first visit MC:
                #     it will just consider the returns following first visit to every state 
                if not any(episode[prior_t][1] == s_t and episode[prior_t][2] == a_t for prior_t in range(0, t)):
                    if not (s_t, a_t) in counts:
                        counts[(s_t, a_t)] = 1
                    else:
                        counts[(s_t, a_t)] += 1
                    
                    learning_rate = 1/counts[(s_t,a_t)]
                    Q[s_t, a_t] = Q[s_t, a_t] + learning_rate * (G - Q[s_t, a_t])
                    
                    # e-greedy policy
                    a_opt = np.argmax(Q[s_t,:])
                    for a in range(env.get_action_size()):
                        if a == a_opt:
                            policy[s_t][a] = 1 - E + E/env.get_action_size()
                        else:
                            policy[s_t][a] = E/env.get_action_size()
            
            values.append(np.sum(policy*Q, axis=1))

        return policy, values, total_rewards
