import logging
import os
import numpy as np
import argparse
import warnings
import yaml
from os.path import join as pjoin
import sys
sys.path.append(sys.path[0] + "/..")

import torch
from torch.utils.tensorboard import SummaryWriter
from agent import RLAgent

from helpers.generic import SlidingAverage, to_np
from helpers.generic import get_experiment_dir, dict2list
from helpers.setup_logger import setup_logging, log_git_commit
from test_agent import test
logger = logging.getLogger(__name__)

import gym
import textworld
import textworld.gym


def request_infos():
    """Request the infos the agent expects from the environment

    Returns:
        request_infos: EnvInfos"""
    request_infos = textworld.EnvInfos()
    request_infos.description = True
    request_infos.inventory = True
    request_infos.entities = True
    request_infos.verbs = True
    request_infos.admissible_commands = True
    request_infos.command_templates = True
    request_infos.max_score = True
    request_infos.intermediate_reward = True
    request_infos.objective = True
    request_infos.feedback = True
    return request_infos


def get_commands(commands_files):
    commands = []
    for command_file in commands_files:
        with open(command_file, "r") as fp:
            for line in fp:
                line = line.strip()
                if len(line) > 0:
                    commands.append(line)
    return list(dict.fromkeys(commands))


def get_word_vocab(vocab_file):
    with open(vocab_file) as fp:
        word_vocab = fp.read().split("\n")
    return word_vocab


def train(config):
    # train env
    print('Setting up TextWorld environment...')
    batch_size = config['training']['scheduling']['batch_size']
    
    requested_infos = request_infos()
    game_files = config['general']['game_files']
    env_id = textworld.gym.register_games(game_files,
                                          requested_infos,
                                          batch_size=batch_size,
                                          asynchronous=True, auto_reset=False,
                                          max_episode_steps=50, # used in the original implementation
                                          name="training")
    env = gym.make(env_id)
    env.seed(config['general']['random_seed'])
    print('Done.')

    # Set the random seed manually for reproducibility.
    np.random.seed(config['general']['random_seed'])
    torch.manual_seed(config['general']['random_seed'])
    if torch.cuda.is_available():
        if not config['general']['use_cuda']:
            logger.warning("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
        else:
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(config['general']['random_seed'])
    else:
        config['general']['use_cuda'] = False  # Disable CUDA.
        
    revisit_counting = config['general']['revisit_counting']
    replay_batch_size = config['general']['replay_batch_size']
    replay_memory_capacity = config['general']['replay_memory_capacity']
    replay_memory_priority_fraction = config['general']['replay_memory_priority_fraction']

    vocab_file = config["general"]["vocab_file"]
    word_vocab = get_word_vocab(vocab_file)
    word2id = {}
    for i, w in enumerate(word_vocab):
        word2id[w] = i
        
    
    commands_files = config["general"]["commands_files"]
    commands = get_commands(commands_files)

    agent = RLAgent(config, word_vocab, commands,
                    replay_memory_capacity=replay_memory_capacity, replay_memory_priority_fraction=replay_memory_priority_fraction)

    init_learning_rate = config['training']['optimizer']['learning_rate']

    exp_dir = get_experiment_dir(config)
    summary = SummaryWriter(exp_dir)

    parameters = filter(lambda p: p.requires_grad, agent.model.parameters())
    if config['training']['optimizer']['step_rule'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=init_learning_rate)
    elif config['training']['optimizer']['step_rule'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=init_learning_rate)

    log_every = 100
    reward_avg = SlidingAverage('reward avg', steps=log_every)
    step_avg = SlidingAverage('step avg', steps=log_every)
    loss_avg = SlidingAverage('loss avg', steps=log_every)
    score_avg = SlidingAverage('scores avgs', steps=log_every)

    # save & reload checkpoint only in 0th agent
    best_avg_reward = -10000
    best_avg_step = 10000

    # step penalty
    discount_gamma = config['general']['discount_gamma']
    provide_prev_action = config['general']['provide_prev_action']

    # epsilon greedy
    epsilon_anneal_epochs = config['general']['epsilon_anneal_epochs']
    epsilon_anneal_from = config['general']['epsilon_anneal_from']
    epsilon_anneal_to = config['general']['epsilon_anneal_to']

    # counting reward
    revisit_counting_lambda_anneal_epochs = config['general']['revisit_counting_lambda_anneal_epochs']
    revisit_counting_lambda_anneal_from = config['general']['revisit_counting_lambda_anneal_from']
    revisit_counting_lambda_anneal_to = config['general']['revisit_counting_lambda_anneal_to']

    epsilon = epsilon_anneal_from
    revisit_counting_lambda = revisit_counting_lambda_anneal_from
    
    max_training_steps = config['training']['scheduling']['training_steps']
    training_steps = 0
    epoch = 0
    
    while training_steps < max_training_steps:
    # for epoch in range(config['training']['scheduling']['epoch']):
        agent.model.train()
        obs, infos = env.reset()
        agent.reset(infos)
        print_command_string, print_rewards = [[] for _ in obs], [[] for _ in obs]
        print_interm_rewards = [[] for _ in obs]
        print_rc_rewards = [[] for _ in obs]

        dones = [False] * batch_size
        rewards = None
        
        scores = np.array([0] * len(obs))
        max_scores = np.array(infos["max_score"])
        
        avg_loss_in_this_game = []

        new_observation_strings = agent.get_observation_strings(infos)
        if revisit_counting:
            agent.reset_binarized_counter(batch_size)
            revisit_counting_rewards = agent.get_binarized_count(new_observation_strings)

        current_game_step = 0
        prev_actions = ["" for _ in range(batch_size)] if provide_prev_action else None
        input_description, description_id_list = agent.get_game_step_info(obs, infos, prev_actions)

        while not all(dones):

            c_idx, chosen_strings, state_representation = agent.generate_one_command(input_description, epsilon=epsilon)
            old_scores = scores
            obs, scores, dones, infos = env.step(chosen_strings)
            
            # calculate immediate reward from scores and normalize it
            rewards = (np.array(scores) - old_scores) / max_scores
            rewards = np.array(rewards, dtype=np.float32)
            
            training_steps += sum([int(not finished) for finished in dones])
            
            new_observation_strings = agent.get_observation_strings(infos)
            if provide_prev_action:
                prev_actions = chosen_strings
            # counting
            if revisit_counting:
                revisit_counting_rewards = agent.get_binarized_count(new_observation_strings, update=True)
            else:
                revisit_counting_rewards = [0.0 for _ in range(batch_size)]
            agent.revisit_counting_rewards.append(revisit_counting_rewards)
            revisit_counting_rewards = [float(format(item, ".3f")) for item in revisit_counting_rewards]

            for i in range(len(infos)):
                print_command_string[i].append(chosen_strings[i])
                print_rewards[i].append(rewards[i])
                print_interm_rewards[i].append(infos["intermediate_reward"][i])
                print_rc_rewards[i].append(revisit_counting_rewards[i])
            if type(dones) is bool:
                dones = [dones] * batch_size
            agent.rewards.append(rewards)
            agent.dones.append(dones)
            agent.intermediate_rewards.append(infos["intermediate_reward"])
            agent.scores.append(scores)
            # computer rewards, and push into replay memory
            rewards_np, rewards, mask_np, mask = agent.compute_reward(revisit_counting_lambda=revisit_counting_lambda, revisit_counting=revisit_counting)

            curr_description_id_list = description_id_list
            input_description, description_id_list = agent.get_game_step_info(obs, infos, prev_actions)

            for b in range(batch_size):
                if mask_np[b] == 0:
                    continue
                if replay_memory_priority_fraction == 0.0:
                    # vanilla replay memory
                    agent.replay_memory.push(curr_description_id_list[b], c_idx[b], rewards[b], mask[b], dones[b],
                                             description_id_list[b], new_observation_strings[b])
                else:
                    # prioritized replay memory
                    is_prior = rewards_np[b] > 0.0
                    agent.replay_memory.push(is_prior, curr_description_id_list[b], c_idx[b], rewards[b], mask[b], dones[b],
                                             description_id_list[b], new_observation_strings[b])

            if current_game_step > 0 and current_game_step % config["general"]["update_per_k_game_steps"] == 0:
                policy_loss = agent.update(replay_batch_size, discount_gamma=discount_gamma)
                if policy_loss is None:
                    continue
                loss = policy_loss
                # Backpropagate
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), config['training']['optimizer']['clip_grad_norm'])
                optimizer.step()  # apply gradients
                avg_loss_in_this_game.append(to_np(policy_loss))
            current_game_step += 1
            
            if training_steps >= max_training_steps:
                break

        agent.finish()
        avg_loss_in_this_game = np.mean(avg_loss_in_this_game)
        reward_avg.add(agent.final_rewards.mean())
        step_avg.add(agent.step_used_before_done.mean())
        loss_avg.add(avg_loss_in_this_game)
        score_avg.add(agent.final_scores.mean())
        
        # annealing
        if epoch < epsilon_anneal_epochs:
            epsilon -= (epsilon_anneal_from - epsilon_anneal_to) / float(epsilon_anneal_epochs)
        if epoch < revisit_counting_lambda_anneal_epochs:
            revisit_counting_lambda -= (revisit_counting_lambda_anneal_from - revisit_counting_lambda_anneal_to) / float(revisit_counting_lambda_anneal_epochs)

        # Tensorboard logging #
        # (1) Log some numbers
        summary.add_scalar('avg_reward', reward_avg.value, training_steps)
        summary.add_scalar('curr_reward', agent.final_rewards.mean(), training_steps)
        summary.add_scalar('curr_interm_reward', agent.final_intermediate_rewards.mean(), training_steps)
        summary.add_scalar('curr_counting_reward', agent.final_counting_rewards.mean(), training_steps)
        summary.add_scalar('avg_step', step_avg.value, training_steps)
        summary.add_scalar('curr_step', agent.step_used_before_done.mean(), training_steps)
        summary.add_scalar('loss_avg', loss_avg.value, training_steps)
        summary.add_scalar('curr_loss', avg_loss_in_this_game, training_steps)
        summary.add_scalar('avg_score', score_avg.value / max_scores[0], training_steps)
        summary.add_scalar('curr_score', agent.final_scores.mean() / max_scores[0], training_steps)


        msg = 'E#{:03d}, TS#{}, R={:.3f}/{:.3f}/IR{:.3f}/CR{:.3f}, Score={:.3f}/{:.3f}, S={:.3f}/{:.3f}, L={:.3f}/{:.3f}, epsilon={:.4f}, lambda_counting={:.4f}'
        msg = msg.format(epoch, training_steps,
                         np.mean(reward_avg.value), agent.final_rewards.mean(), agent.final_intermediate_rewards.mean(), agent.final_counting_rewards.mean(),
                         score_avg.value / max_scores[0], agent.final_scores.mean() / max_scores[0],
                         np.mean(step_avg.value), agent.step_used_before_done.mean(),
                         np.mean(loss_avg.value), avg_loss_in_this_game,
                         epsilon, revisit_counting_lambda)
        print(msg)
        epoch += 1


if __name__ == '__main__':
    for _p in ['saved_models']:
        if not os.path.exists(_p):
            os.mkdir(_p)
    parser = argparse.ArgumentParser(description="train network.")
    parser.add_argument("-c", "--config_dir", default='config', help="the default config directory")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-vv", "--very-verbose", help="print out warnings", action="store_true")
    args = parser.parse_args()

    if args.very_verbose:
        args.verbose = args.very_verbose
        warnings.simplefilter("default", textworld.TextworldGenerationWarning)

    # Read config from yaml file.
    config_file = pjoin(args.config_dir, 'config.yaml')
    with open(config_file) as reader:
        config = yaml.safe_load(reader)

    default_logs_path = get_experiment_dir(config)
    setup_logging(default_config_path=pjoin(args.config_dir, 'logging_config.yaml'),
                  default_level=logging.INFO, add_time_stamp=True,
                  default_logs_path=default_logs_path)
    log_git_commit(logger)

    train(config=config)
