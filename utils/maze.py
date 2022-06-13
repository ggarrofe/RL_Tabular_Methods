import numpy as np
import random
import matplotlib.pyplot as plt

class GraphicsMaze(object):

    def __init__(self, shape, locations, default_reward, obstacle_locs, absorbing_locs, absorbing_rewards, absorbing):

        self.shape = shape
        self.locations = locations
        self.absorbing = absorbing

        # Walls
        self.walls = np.zeros(self.shape)
        for ob in obstacle_locs:
            self.walls[ob] = 20

        # Rewards
        self.rewarders = np.ones(self.shape) * default_reward
        for i, rew in enumerate(absorbing_locs):
            self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

        # Print the map to show it
        self.paint_maps()

    def paint_maps(self):
        """
        Print the Maze topology (obstacles, absorbing states and rewards)
        input: /
        output: /
        """
        plt.figure(figsize=(15, 10))
        plt.imshow(self.walls + self.rewarders)
        plt.show()

    def paint_state(self, state):
        """
        Print one state on the Maze topology (obstacles, absorbing states and rewards)
        input: /
        output: /
        """
        states = np.zeros(self.shape)
        states[state] = 30
        plt.figure(figsize=(15, 10))
        plt.imshow(self.walls + self.rewarders + states)
        plt.show()

    def draw_deterministic_policy(self, Policy, path):
        """
        Draw a deterministic policy
        input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
        output: /
        """
        plt.figure(figsize=(15, 10))
        plt.imshow(self.walls + self.rewarders)  # Create the graph of the Maze
        for state, action in enumerate(Policy):
            # If it is an absorbing state, don't plot any action
            if(self.absorbing[0, state]):
                continue
            # List of arrows corresponding to each possible action
            arrows = [r"$\uparrow$", r"$\rightarrow$",
                      r"$\downarrow$", r"$\leftarrow$"]
            action_arrow = arrows[action]  # Take the corresponding action
            location = self.locations[state]  # Compute its location on graph
            plt.text(location[1], location[0], action_arrow,
                     ha='center', va='center')  # Place it on graph
            
        plt.savefig(path+"policy_grid.png", dpi=150, bbox_inches='tight')
        plt.show()

    def draw_policy(self, Policy, path):
        """
        Draw a policy (draw an arrow in the most probable direction)
        input: Policy {np.array} -- policy to draw as probability
        output: /
        """
        deterministic_policy = np.array(
            [np.argmax(Policy[row, :]) for row in range(Policy.shape[0])])
        self.draw_deterministic_policy(deterministic_policy, path)

    def draw_value(self, Value, path):
        """
        Draw a policy value
        input: Value {np.array} -- policy values to draw
        output: /
        """
        plt.figure(figsize=(15, 10))
        plt.imshow(self.walls + self.rewarders)  # Create the graph of the Maze
        for state, value in enumerate(Value):
            if(self.absorbing[0, state]):  # If it is an absorbing state, don't plot any value
                continue
            # Compute the value location on graph
            location = self.locations[state]
            plt.text(location[1], location[0], round(value, 2),
                     ha='center', va='center')  # Place it on graph
            
        plt.savefig(path+"value_grid.png", dpi=150, bbox_inches='tight')
        plt.show()

    def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
        """
        Draw a grid representing multiple deterministic policies
        input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
        output: /
        """
        plt.figure(figsize=(20, 8))
        for subplot in range(len(Policies)):  # Go through all policies
            # Create a subplot for each policy
            ax = plt.subplot(n_columns, n_lines, subplot+1)
            # Create the graph of the Maze
            ax.imshow(self.walls+self.rewarders)
            for state, action in enumerate(Policies[subplot]):
                # If it is an absorbing state, don't plot any action
                if(self.absorbing[0, state]):
                    continue
                # List of arrows corresponding to each possible action
                arrows = [r"$\uparrow$", r"$\rightarrow$",
                          r"$\downarrow$", r"$\leftarrow$"]
                action_arrow = arrows[action]  # Take the corresponding action
                # Compute its location on graph
                location = self.locations[state]
                plt.text(location[1], location[0], action_arrow,
                         ha='center', va='center')  # Place it on graph
            # Set the title for the graph given as argument
            ax.title.set_text(title[subplot])
            
        plt.show()

    def draw_policy_grid(self, Policies, title, n_columns, n_lines):
        """
        Draw a grid representing multiple policies (draw an arrow in the most probable direction)
        input: Policy {np.array} -- array of policies to draw as probability
        output: /
        """
        deterministic_policies = np.array(
            [[np.argmax(Policy[row, :]) for row in range(Policy.shape[0])] for Policy in Policies])
        self.draw_deterministic_policy_grid(
            deterministic_policies, title, n_columns, n_lines)

    def draw_value_grid(self, Values, title, n_columns, n_lines):
        """
        Draw a grid representing multiple policy values
        input: Values {np.array of np.array} -- array of policy values to draw
        output: /
        """
        plt.figure(figsize=(20, 8))
        for subplot in range(len(Values)):  # Go through all values
            # Create a subplot for each value
            ax = plt.subplot(n_columns, n_lines, subplot+1)
            # Create the graph of the Maze
            ax.imshow(self.walls+self.rewarders)
            for state, value in enumerate(Values[subplot]):
                # If it is an absorbing state, don't plot any value
                if(self.absorbing[0, state]):
                    continue
                # Compute the value location on graph
                location = self.locations[state]
                plt.text(location[1], location[0], round(
                    value, 1), ha='center', va='center')  # Place it on graph
            # Set the title for the graoh given as argument
            ax.title.set_text(title[subplot])
        
        plt.show()


class Maze(object):

    # [Action required]
    def __init__(self):
        """
        Maze initialisation.
        input: /
        output: /
        """

        self._prob_success = 0.8  # float
        self._gamma = 0.98  # float
        self._goal = 1  # integer (0 for R0, 1 for R1, 2 for R2, 3 for R3)

        # Build the maze
        self._build_maze()

    # Functions used to build the Maze environment
    # You DO NOT NEED to modify them

    def _build_maze(self):
        """
        Maze initialisation.
        input: /
        output: /
        """

        # Properties of the maze
        self._shape = (13, 10)
        self._obstacle_locs = [
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 7), (1, 8), (1, 9),
            (2, 1), (2, 2), (2, 3), (2, 7),
            (3, 1), (3, 2), (3, 3), (3, 7),
            (4, 1), (4, 7),
            (5, 1), (5, 7),
            (6, 5), (6, 6), (6, 7),
            (8, 0),
            (9, 0), (9, 1), (9, 2), (9, 6), (9, 7), (9, 8), (9, 9),
            (10, 0)
        ]  # Location of obstacles
        # Location of absorbing states
        self._absorbing_locs = [(2, 0), (2, 9), (10, 1), (12, 9)]
        self._absorbing_rewards = [
            (500 if (i == self._goal) else -50) for i in range(4)] # Reward of absorbing states
        self._starting_locs = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                               (0, 6), (0, 7), (0, 8), (0, 9)]  
        self._default_reward = -1  # Reward for each action performs in the environment
        self._max_t = 500  # Max number of steps in the environment

        # Actions
        self._action_size = 4
        # Direction 0 is 'N', 1 is 'E' and so on
        self._direction_names = ['N', 'E', 'S', 'W']

        # States
        self._locations = []
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                loc = (i, j)
                # Adding the state to locations if it is no obstacle
                if self._is_location(loc):
                    self._locations.append(loc)
        self._state_size = len(self._locations)

        # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
        self._neighbours = np.zeros((self._state_size, 4))

        for state in range(self._state_size):
            loc = self._get_loc_from_state(state)

            # North
            neighbour = (loc[0]-1, loc[1])  # North neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][self._direction_names.index(
                    'N')] = self._get_state_from_loc(neighbour)
            else:  # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index(
                    'N')] = state

            # East
            neighbour = (loc[0], loc[1]+1)  # East neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][self._direction_names.index(
                    'E')] = self._get_state_from_loc(neighbour)
            else:  # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index(
                    'E')] = state

            # South
            neighbour = (loc[0]+1, loc[1])  # South neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][self._direction_names.index(
                    'S')] = self._get_state_from_loc(neighbour)
            else:  # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index(
                    'S')] = state

            # West
            neighbour = (loc[0], loc[1]-1)  # West neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][self._direction_names.index(
                    'W')] = self._get_state_from_loc(neighbour)
            else:  # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index(
                    'W')] = state

        # Absorbing
        self._absorbing = np.zeros((1, self._state_size))
        for a in self._absorbing_locs:
            absorbing_state = self._get_state_from_loc(a)
            self._absorbing[0, absorbing_state] = 1

        # Transition matrix
        # Empty matrix of domension S*S*A
        self._T = np.zeros(
            (self._state_size, self._state_size, self._action_size))
        for action in range(self._action_size):
            for outcome in range(4):  # For each direction (N, E, S, W)
                # The agent has prob_success probability to go in the correct direction
                if action == outcome:
                    # (theoritically equal to self.prob_success but avoid rounding error and garanty a sum of 1)
                    prob = 1 - 3.0 * ((1.0 - self._prob_success) / 3.0)
                # Equal probability to go into one of the other directions
                else:
                    prob = (1.0 - self._prob_success) / 3.0

                # Write this probability in the transition matrix
                for prior_state in range(self._state_size):
                    # If absorbing state, probability of 0 to go to any other states
                    if not self._absorbing[0, prior_state]:
                        # Post state number
                        post_state = self._neighbours[prior_state, outcome]
                        # Transform in integer to avoid error
                        post_state = int(post_state)
                        self._T[prior_state, post_state, action] += prob

        # Reward matrix
        self._R = np.ones((self._state_size, self._state_size,
                          self._action_size))  # Matrix filled with 1
        self._R = self._default_reward * self._R  # Set default_reward everywhere
        for i in range(len(self._absorbing_rewards)):  # Set absorbing states rewards
            post_state = self._get_state_from_loc(self._absorbing_locs[i])
            self._R[:, post_state, :] = self._absorbing_rewards[i]

        # Creating the graphical Maze world
        self._graphics = GraphicsMaze(self._shape, self._locations, self._default_reward,
                                      self._obstacle_locs, self._absorbing_locs, self._absorbing_rewards, self._absorbing)

        # Reset the environment
        self.reset()

    def _is_location(self, loc):
        """
        Is the location a valid state (not out of Maze and not an obstacle)
        input: loc {tuple} -- location of the state
        output: _ {bool} -- is the location a valid state
        """
        if (loc[0] < 0 or loc[1] < 0 or loc[0] > self._shape[0]-1 or loc[1] > self._shape[1]-1):
            return False
        elif (loc in self._obstacle_locs):
            return False
        else:
            return True

    def _get_state_from_loc(self, loc):
        """
        Get the state number corresponding to a given location
        input: loc {tuple} -- location of the state
        output: index {int} -- corresponding state number
        """
        return self._locations.index(tuple(loc))

    def _get_loc_from_state(self, state):
        """
        Get the state number corresponding to a given location
        input: index {int} -- state number
        output: loc {tuple} -- corresponding location
        """
        return self._locations[state]

    # Getter functions used only for DP agents
    def get_T(self):
        return self._T

    def get_R(self):
        return self._R

    def get_absorbing(self):
        return self._absorbing

    # Getter functions used for DP, MC and TD agents
    def get_graphics(self):
        return self._graphics

    def get_action_size(self):
        return self._action_size

    def get_state_size(self):
        return self._state_size

    def get_gamma(self):
        return self._gamma

    # Functions used to perform episodes in the Maze environment
    def reset(self):
        """
        Reset the environment state to one of the possible starting states
        input: /
        output: 
          - t {int} -- current timestep
          - state {int} -- current state of the envionment
          - reward {int} -- current reward
          - done {bool} -- True if reach a terminal state / 0 otherwise
        """
        self._t = 0
        self._state = self._get_state_from_loc(
            self._starting_locs[random.randrange(len(self._starting_locs))])
        self._reward = 0
        self._done = False
        return self._t, self._state, self._reward, self._done

    def step(self, action):
        """
        Perform an action in the environment
        input: action {int} -- action to perform
        output: 
          - t {int} -- current timestep
          - state {int} -- current state of the envionment
          - reward {int} -- current reward
          - done {bool} -- True if reach a terminal state / 0 otherwise
        """

        # If environment already finished, print an error
        if self._done or self._absorbing[0, self._state]:
            print("Please reset the environment")
            return self._t, self._state, self._reward, self._done

        # Drawing a random number used for probaility of next state
        probability_success = random.uniform(0, 1)

        # Look for the first possible next states (so get a reachable state even if probability_success = 0)
        new_state = 0
        while self._T[self._state, new_state, action] == 0:
            new_state += 1
        assert self._T[self._state, new_state,
                       action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

        # Find the first state for which probability of occurence matches the random value
        total_probability = self._T[self._state, new_state, action]
        while (total_probability < probability_success) and (new_state < self._state_size-1):
            new_state += 1
            total_probability += self._T[self._state, new_state, action]
        assert self._T[self._state, new_state,
                       action] != 0, "Selected state should be probability 0, something might be wrong in the environment."

        # Setting new t, state, reward and done
        self._t += 1
        self._reward = self._R[self._state, new_state, action]
        self._done = self._absorbing[0, new_state] or self._t > self._max_t
        self._state = new_state
        return self._t, self._state, self._reward, self._done
