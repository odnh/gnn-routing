import time
import torch
from rlpyt.agents.pg.cp import CpFfAgent
from rlpyt.envs.gym import make as gym_make

data = torch.load("params.pkl")
cpa = CpFfAgent(initial_model_state_dict=data['agent_state_dict'])
env = gym_make(id="CartPole-v0")
cpa.initialize(env.spaces)

done = False
env.seed(1234)
obs = env.reset()
#env.render()
obs_list = [obs]
act_list = []
while not done:
    action = torch.argmax(cpa.model.pi(torch.tensor(obs))).numpy()
    data = env.step(action)
    obs = data.observation
    done = data.done
    obs_list.append(obs)
    act_list.append(action)

torch.save(obs_list, "obs_list")
torch.save(act_list, "act_list")
env.close()
