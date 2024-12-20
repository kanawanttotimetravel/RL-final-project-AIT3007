from magent2.environments import battle_v4
import os
import cv2

import torch

from torch_model import QNetwork
from final_torch_model import QNetwork as FinalQNetwork

def run_game(env, red_policy, blue_policy):
    frames = []
    env.reset()
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "red":
                action = red_policy(env, agent, observation)
            else:
                action = blue_policy(env, agent, observation)

        env.step(action)

        if agent == "red_0":
            frames.append(env.render())
    return frames


def frames_to_video(frames, vid_dir, vid_name="dummy"):
    height, width, _ = frames[0].shape
    fps = 35
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"{vid_name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()


def simulation(env, red_policy, blue_policy, vid_dir="video", vid_name="dummy"):
    frames = run_game(env, red_policy, blue_policy)
    frames_to_video(frames, vid_dir, vid_name)


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


if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    observation_space = env.observation_space("red_0").shape
    action_space = env.action_space("red_0").n

    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)

    red_network = load_network('red.pt', QNetwork, observation_space, action_space)
    red_final_network = load_network('red_final.pt', FinalQNetwork, observation_space, action_space)
    blue_network = load_network('blue.pt', FinalQNetwork, observation_space, action_space)

    random_policy = get_policy()
    red_policy = get_policy(red_network)
    red_final_policy = get_policy(red_final_network)
    blue_policy = get_policy(blue_network)

    simulation(env, random_policy, blue_policy, vid_name="random-test")
    print('Done random')
    simulation(env, red_policy, blue_policy, vid_name="red-test")
    print('Done red') 
    simulation(env, red_final_policy, blue_policy, vid_name="red-final-test")
    print('Done red final')