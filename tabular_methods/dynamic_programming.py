import numpy as np

# This class define the Dynamic Programing agent

class DP_agent(object):

    def solve(self, env):
        """
        Solve a given Maze environment using Dynamic Programming
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - V {np.array} -- Corresponding value function 
        """

        THRESHOLD = 0.0001
        policy = np.zeros((env.get_state_size(), env.get_action_size()))
        V = np.zeros(env.get_state_size())
        delta = THRESHOLD  
        
        epochs = 0
        while delta >= THRESHOLD:
            epochs += 1
            delta = 0.0
            for o_s in range(env.get_state_size()):
                if not env.get_absorbing()[0, o_s]:
                    opt_v, opt_a = float('-inf'), 0

                    for a in range(env.get_action_size()):
                        v = sum(env.get_T()[o_s,:,a] * (env.get_R()[o_s,:,a] + env.get_gamma() * V[:]))
                        if v > opt_v:
                            opt_v, opt_a = v, a
                    
                    delta = max(delta, np.absolute(opt_v - V[o_s]))
                    V[o_s] = opt_v
                    policy[o_s] *= 0.0
                    policy[o_s][opt_a] = 1.0
                    
        return policy, V, epochs
