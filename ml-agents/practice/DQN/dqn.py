import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
import EngineConfigurationChannel

state_size = [3*2, 64, 84]
action_size = 4

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 10000
discount_factor = 0.9
learning_rate = 2.5e-4

run_step = 50000 if train_mode else 0
test_step = 5000
train_start_step = 5000
target_update_step = 500

print_interval = 10
save_interval = 100

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else 0
epsilon_min = 0.1
explore_step = run_step * 0.8
epsilon_delta = (epsilon_init - epsilon_min) / explore_step if train_mode else 0

VISUAL_OBS = 0
GOAL_OBS = 1
VECTOR_OBS = 2
OBS = VISUAL_OBS

game = "GridWorld"
os_name = platform.system()
if os_name == 'Windows':
  env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
  env_name = f"../envs/{game}_{os_name}"

date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/DQN/{date_time}"
load_path = f"./saved_models{game}/DQN/20210514201212"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(torch.nn.Module):
    def __init__(self, ** kwargs):
        super(DQN, self).__init__(**kwargs)
        self.conv1 = torch.nn.Conv2d(in_channels = state_size[0], out_channels = 32, kernel_size = 8, stride = 4)
        dim1 = ((state_size[1] - 8) // 4 + 1, (state_size[2] - 8) // 4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernal_size = 4, stride = 2)
        dim2 = ((dim1[0] - 4) // 2 + 1, (dim1[1] - 4) // 2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1)
        dim3 = ((dim2[0] - 3) // 1 + 1, (dim2[1] - 3) // 1 + 1)

        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * dim3[0] * dim3[1], 512)
        self.q = torch.nn.Linear(512, action_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        return self.q(x)

class DQNAgent:
    def __init__(self):
        self.network = DQN().to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen = mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)

        if load_model:
            print(f"... Load Model from {load_path}/ckpt")
            checkpoint = torch.load(load_path + '/ckpt', map_location = device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def get_action(self, state, training = True):
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval

        if epsilon > random.random():
            action = np.random.randint(0, action_size, size = (state.shape[0], 1))
        else:
            q = self.network(torch.FloatTensor(state).to(device))
            action = torch.argmax(q, axis = -1, keepdim = True).date.cpu().numpy()
            return action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state = np.stack([b[0] for b in batch], axis = 0)
        action = np.stack([b[1] for b in batch], axis = 0)
        reward = np.stack([b[2] for b in batch], axis = 0)
        next_state = np.stack([b[3] for b in batch], axis = 0)
        done = np.stack([b[4] for b in batch], axis = 0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), [state, action, reward, next_state, done])

        eye = torch.eye(action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims = True)

        with torch.no_grad():
            next_q = self.target_network(next_state)
            target_q = reward + next_q.max(1, keepdims = True).values * ((1 - done) * discount_factor)

            loss = F.smooth_l1_loss(q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta)

            return loss.item()

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path + '/ckpt')

    def write_summary(self, score, loss, epsilon, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/loss", loss, step)
        self.writer.add_scalar("model/epsilon", epsilon, step)

if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec, term = env.get_steps(behavior_name)

    agent = DQNAgent()
    losses, scores, episode, score = [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        preprocess = lambda obs, goal: np.concatenate((obs * goal[0][0], obs * goal[0][1]), axis = -1)
        state = preprocess(dec.obs[OBS], dec.obs[GOAL_OBS])
        action = agent.get_action(state, train_mode)
        real_action = action + 1
        action_tuple = ActionTuple()
        action_tuple.add_discrete(real_action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        next_state = preprocess(term.obs[OBS], term.obs[GOAL_OBS]) if done\
            else preprocess(dec.obs[OBS], dec.obs[GOAL_OBS])
        score += reward[0]

        if train_mode:
            agent.append_sample(state[0], action[0], reward, next_state[0], [done])

        if train_mode and step > max(batch_size, train_start_step):
            loss = agent.train_model()
            losses.append(loss)

            if step % target_update_step == 0:
                agent.update_target()

        if done:
            episode += 1
            scores.append(score)
            score = 0

            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_loss = np.mean(losses)
                agent.write_summary(mean_score, mean_loss, agent.epsilon, step)
                losses, scores = [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                      f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()
