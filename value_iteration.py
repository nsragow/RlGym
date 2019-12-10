import gym


class ValueModel:

    def __init__(self, env, gamma=.9):
        if (not gamma < 1) or (gamma <= 0): # if gamma is not between 0 and 1
            raise ValueError(f"gamma {gamma} does not fall between 0 and 1")
        self.gamma = gamma
        self.transition = env.env.P
        self.states = [0] * len(self.transition.keys())
        self.cutoff = .099
        self.converge_cutoff = .0001

    def value(self, state: int):
        if state >= len(self.state_count):
            raise ValueError(f"Index {state} out of range. The length of states is {len(self.states)}")
        elif state < 0:
            raise ValueError("Index {state} less than zero")
        return self.states[state]

    def full_reevaluation(self):
        all_done = False
        while not all_done:
            all_done = True
            for state in range(len(self.states)):
                if not self.reevaluate_value(state):
                    all_done = False

    def reevaluate_value(self, state):
        new_value = self._evaluate(state)
        if new_value == float("inf"):
            raise ValueError("Gradients exploded, discount improperly set up leading to inf values")
        old_value = self.states[state]
        self.states[state] = new_value
        # return true if this states change is small
        return abs(new_value - old_value) < self.converge_cutoff

    def _evaluate(self, state):
        actions_at_state = self.transition[state]
        expected_rewards = []
        for action in actions_at_state.keys():  # need to find the max action
            total_reward = 0
            transitions = actions_at_state[action]

            # add value for going to all possible states multiplied by the chance to go there
            for transition in transitions:
                chance, new_state, reward, done = transition
                this_actions_total_reward = 0
                this_actions_total_reward += reward + self.states[new_state]
                total_reward += chance * this_actions_total_reward
            # now that this action had all its reward calculated, add it to the
            # expected_rewards list for consideration as the max
            expected_rewards.append(self.gamma * total_reward)
        return max(expected_rewards)

    def extract_policy(self):
        policy = [0] * len(self.states)

        for state in range(len(self.states)):
            best_action = 0
            current_reward = 0

            actions_at_state = self.transition[state]

            for action in actions_at_state.keys():  # need to find the max action
                reward = 0
                transitions = actions_at_state[action]
                # add value for going to all possible states multiplied by the chance to go there
                for transition in transitions:
                    chance, new_state, reward, done = transition
                    reward += chance * (reward + self.states[new_state])
                if current_reward < reward:
                    current_reward = reward
                    best_action = action
            policy[state] = best_action
        return policy





if __name__ == "__main__":
    frozen_lake = gym.make("FrozenLake-v0")
    model = ValueModel(frozen_lake)
    iteration = 0
    model.full_reevaluation()
    print(model.states)
    print(model.extract_policy())


