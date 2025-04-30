# ppo_run.py
import time
import numpy as np
import os
import torch
from scripts.utils.logger import Logger
from scripts.utils import pytorch_utils as ptu
from scripts.utils.utils import rollout_trajectories, compute_metrics, convert_listofrollouts
from scripts.agents.ppo_agent import PPOAgent
from scripts.utils.virtual_dynamics_handler import VirtualDynamicsHandler
from argparse import ArgumentParser


def run_ppo_training(args):
    # 로그 디렉토리 생성
    logdir = os.path.join("results", args.exp_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S"))
    os.makedirs(logdir, exist_ok=True)
    logger = Logger(logdir)

    if args.seed:
        # 랜덤 시드
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

        # 환경 초기화
        env = VirtualDynamicsHandler(max_episode_size=args.max_ep_len)
    else:
        env = VirtualDynamicsHandler(max_episode_size=args.max_ep_len)

    # obs, action dimension
    ob_dim = np.prod(env.grid_shape)  # 17 x 10 = 170
    ac_dim = len(env.all_actions)     # 전체 valid 사각형 개수

    # PPO 에이전트 초기화
    agent = PPOAgent(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
        normalize_advantages=args.normalize_advantages,
        n_ppo_epochs=args.n_ppo_epochs,
        n_ppo_minibatches=args.n_ppo_minibatches,
        ppo_cliprange=args.ppo_cliprange,
    )

    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")

        # Trajectory 샘플링
        trajs, envsteps_this_batch = rollout_trajectories(
            env, agent.actor, args.batch_size, args.max_ep_len, render=False
        )
        total_envsteps += envsteps_this_batch

        # 트라젝토리에서 필요한 데이터 추출
        obs, actions, next_obs, terminals, rewards, rewards_per_traj = convert_listofrollouts(trajs)

        # 에이전트 업데이트
        train_info = agent.update(obs, actions, rewards_per_traj, terminals)

        if itr % args.log_freq == 0:
            logs = compute_metrics(trajs, trajs)
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            print("Logging...")
            for k, v in logs.items():
                (f"{k}: {v}")
                logger.log_scalar(v, k, itr)
            logger.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ppo_virtual")
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--max_ep_len", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--baseline_learning_rate", type=float, default=1e-3)
    parser.add_argument("--baseline_gradient_steps", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--layer_size", type=int, default=64)
    parser.add_argument("--normalize_advantages", action="store_true")
    parser.add_argument("--n_ppo_epochs", type=int, default=4)
    parser.add_argument("--n_ppo_minibatches", type=int, default=4)
    parser.add_argument("--ppo_cliprange", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", type=int, default=0)
    parser.add_argument("--log_freq", type=int, default=1)

    args = parser.parse_args()
    run_ppo_training(args)
