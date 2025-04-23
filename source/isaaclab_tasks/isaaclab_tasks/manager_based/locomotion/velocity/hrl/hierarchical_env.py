# hierarchical_env.py
class HierarchicalLocomotionEnv:
    def __init__(self, base_env, manager_policy, flat_policy, rough_policy):
        self.base_env = base_env
        self.manager_policy = manager_policy
        self.flat_policy = flat_policy
        self.rough_policy = rough_policy

    def step(self, action=None):
        obs = self.base_env.get_observation()
        height_scan = obs['height_scan']
        policy_idx = self.manager_policy.select_policy(height_scan)
        if policy_idx == 0:
            action = self.flat_policy.act(obs)
        else:
            action = self.rough_policy.act(obs)
        next_obs, reward, done, info = self.base_env.step(action)
        info['selected_policy'] = policy_idx
        return next_obs, reward, done, info