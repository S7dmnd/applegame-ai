# inference.py
import torch
import numpy as np
from scripts.networks.ppo_actor import PPOActor
from scripts.agents.simple_digit_classifier import SimpleDigitClassifier
from scripts.utils.dynamics_handler import DynamicsHandler  # 너가 만든 실제 환경
from scripts.utils import pytorch_utils as ptu


def main(cnn_model_path="cnn_model_weights.pth"):

    #CNN 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = SimpleDigitClassifier().to(device)
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.eval()
    # 환경 생성
    env = DynamicsHandler(model=cnn_model, max_episode_size=150)
    obs_shape = (1, 17, 10)
    obs_dim = np.prod(obs_shape) #170
    act_dim = 4  # Continuous box action: (x1, y1, x2, y2)

    # 모델 생성
    actor = PPOActor(obs_dim=obs_dim, act_dim=act_dim)
    actor.load_state_dict(torch.load("ppo_actor_2.pt", map_location=device))
    actor.eval()

    obs = env.reset()  # torch.Tensor, shape: (1, 1, 17, 10)
    done = False
    total_reward = 0
    step = 0

    while not done:
        action = actor.get_action(obs)
        obs, reward, done = env.step(action)
        total_reward += reward
        step += 1

        print(f"[Step {step}] Action: {action}, Reward: {reward}")
        # env.render()  # optional 시각화

    print(f"\n✅ Episode finished in {step} steps | Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
