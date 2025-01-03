from magent2.environments import battle_v4
from torch_model import QNetwork
from final_torch_model import QNetwork as FinalQNetwork
import torch
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # Fallback: tqdm becomes a no-op

def get_policy(q_network=None):
    def random_action(env, agent, observation):
        return env.action_space(agent).sample()
    
    def get_action(env, agent, observation):
        observation = (
            torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
        )
        with torch.no_grad():
            q_values = q_network(observation)
        action = torch.argmax(q_values, dim=1).numpy()[0]

        return action
    if q_network is None:
        return random_action 
    
    return get_action

def load_network(path, architecture, observation_space, action_space):
    model = architecture(observation_space, action_space)
    model.load_state_dict(
        torch.load(path, weights_only=True, map_location="cpu")
    )
    return model

def eval():
    max_cycles = 300
    env = battle_v4.env(map_size=45, max_cycles=max_cycles)
    observation_space = env.observation_space("red_0").shape
    action_space = env.action_space("red_0").n

    device = "cuda" if torch.cuda.is_available() else "cpu"

    red_network = load_network('red.pt', QNetwork, observation_space, action_space)
    red_final_network = load_network('red_final.pt', FinalQNetwork, observation_space, action_space)
    blue_network = load_network('blue.pt', FinalQNetwork, observation_space, action_space)

    random_policy = get_policy()
    red_policy = get_policy(red_network)
    red_final_policy = get_policy(red_final_network)
    blue_policy = get_policy(blue_network)


    def run_eval(env, red_policy, blue_policy, n_episode: int = 100):
        red_win, blue_win = [], []
        red_tot_rw, blue_tot_rw = [], []
        n_agent_each_team = len(env.env.action_spaces) // 2

        for _ in tqdm(range(n_episode)):
            env.reset()
            n_kill = {"red": 0, "blue": 0}
            red_reward, blue_reward = 0, 0

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                agent_team = agent.split("_")[0]

                n_kill[agent_team] += (
                    reward > 4.5
                )  # This assumes default reward settups
                if agent_team == "red":
                    red_reward += reward
                else:
                    blue_reward += reward

                if termination or truncation:
                    action = None  # this agent has died
                else:
                    if agent_team == "red":
                        action = red_policy(env, agent, observation)
                    else:
                        action = blue_policy(env, agent, observation)

                env.step(action)

            who_wins = "red" if n_kill["red"] >= n_kill["blue"] + 5 else "draw"
            who_wins = "blue" if n_kill["red"] + 5 <= n_kill["blue"] else who_wins
            red_win.append(who_wins == "red")
            blue_win.append(who_wins == "blue")

            red_tot_rw.append(red_reward / n_agent_each_team)
            blue_tot_rw.append(blue_reward / n_agent_each_team)

        return {
            "winrate_red": np.mean(red_win),
            "winrate_blue": np.mean(blue_win),
            "average_rewards_red": np.mean(red_tot_rw),
            "average_rewards_blue": np.mean(blue_tot_rw),
        }

    print("=" * 20)
    print("Eval with random policy")
    print(
        run_eval(
            env=env, red_policy=random_policy, blue_policy=blue_policy, n_episode=30
        )
    )
    print("=" * 20)

    print("Eval with trained policy")
    print(
        run_eval(
            env=env, red_policy=red_policy, blue_policy=blue_policy, n_episode=30
        )
    )
    print("=" * 20)

    print("Eval with final trained policy")
    print(
        run_eval(
            env=env,
            red_policy=red_final_policy,
            blue_policy=blue_policy,
            n_episode=30,
        )
    )
    print("=" * 20)


if __name__ == "__main__":
    eval()
