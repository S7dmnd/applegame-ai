import os
import yaml
import argparse

import numpy as np
import torch
import tqdm

import env_configs
from scripts.agents.sac_agent import SoftActorCritic
from scripts.utils.replay_buffer import ReplayBuffer, PERReplayBuffer
from scripts.utils.logger import Logger
from scripts.utils import pytorch_utils as ptu
from scripts.utils import utils
from scripts.scripting_utils import make_logger, make_config


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"] or batch_size

    discrete = False
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        # fps = 1 / env.model.opt.timestep
        fps = 30
    else:
        # fps = env.env.metadata["render_fps"]
        fps = 30

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        args.mlp,
        **config["agent_kwargs"],
    )

    if args.per:
        replay_buffer = PERReplayBuffer(config["replay_buffer_capacity"])
        print("##### Using PER #####")
    else:
        replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation, _ = env.reset()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            # TODO(student): Select an action
            action = agent.get_action(observation)

        # step the environment and add the data to the replay buffer
        next_observation, reward, done, truncated, info = env.step(action)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        )

        if done or truncated:
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation, _ = env.reset()
        else:
            observation = next_observation

        # train SAC agent
        if step >= config["training_starts"]:
            # TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
            # Please refer to cas4160/infrastructure/replay_buffer.py
            batch = replay_buffer.sample(batch_size=config["batch_size"])

            # convert to PyTorch tensors
            batch = ptu.from_numpy(batch)

            # TODO(student): Train the agent using `update` method. `batch` is a dictionary of torch tensors.
            update_info = agent.update(observations=batch["observations"], actions=batch["actions"], rewards=batch["rewards"], next_observations=batch["next_observations"], dones=batch["dones"], step=step)

            # logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            # 마지막 critic update의 td_errors를 꺼내와서 priorities로 사용
            if "indices" in batch:
                # print("PER 잘 적용됐나 확인")
                assert "td_errors" in update_info
                new_priorities = update_info["td_errors"]
                # print("new_priorities:", new_priorities)
                # print("indices:", batch["indices"])
                replay_buffer.update_priorities(batch["indices"], new_priorities)


            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    if k != "td_errors":
                        logger.log_scalar(v, k, step)
                        logger.log_scalars
                logger.flush()

        # run evaluation
        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            logger.log_scalar(step, "Train_EnvstepsSoFar", step)
            logger.log_scalar(np.mean(returns), "Eval_AverageReturn", step)
            print(f"Returns at step {step} : {np.mean(returns)}")

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )
    torch.save(agent.actor.state_dict(), "sac_actor.pt") #저장


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    parser.add_argument("--per", "-per", action="store_true") # PER 켜는 용도
    parser.add_argument("--mlp", "-mlp", action="store_true") # Actor CNN 끄는 용도

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw4_sac_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
