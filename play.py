import os, shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.agent import PEARLAgent
from configs.default import default_config
from launch_experiment import deep_update_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout
from numpy import random

def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=False, save_video=False,sparse_reward=False,specific_context=False,import_z=False,action_error=False,always_uniform=False,
prior_untill_succ=False,high_prob_z=False,keep_succ_z=False,determinstic_z=False,sparse_reward_with_uncertain_information=False):
    '''
    simulate a trained policy adapting to a new task
    optionally save videos of the trajectories - requires ffmpeg

    :variant: experiment configuration dict
    :path_to_exp: path to exp folder
    :num_trajs: number of trajectories to simulate per task (default 1)
    :deterministic: if the policy is deterministic (default stochastic)
    :save_video: whether to generate and save a video (default False)
    '''

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_tasks=list(tasks[-variant['n_eval_tasks']:])
    print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    print ("context encoder== ",context_encoder)
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200,200,200],
        input_size=10,
        output_size=context_encoder,
    )
    
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder_itr_0.pth')))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy_itr_0.pth')))

    # loop through tasks collecting rollouts
    all_rets = []
    video_frames = []
 
    scale_std=1  
    all_trials_returns=[]

    for i in range(10):
        env.reset_task(0)
        agent.clear_z()
        paths = []
        found_suc_traj=False
        number_succ_trajs=0
        first_succ_traj=None
        current_trial_returns=[]

        for n in range(num_trajs):

            if import_z:
                z1=float(input('import z1'))
                z2=float(input('import z2'))
                agent.z=torch.tensor([[z1,z2]])

            path = rollout(env, agent, max_path_length=variant['algo_params']['max_path_length'], accum_context=True, animated=True,scale_std=scale_std,sparse_reward=sparse_reward,specific_context=specific_context)
            trajectory_return= sum(path['rewards'])

            paths.append(path)
    
            if n >1:
                agent.infer_posterior(agent.context,high_prob_z=high_prob_z)

            else:
                agent.clear_z(clear_context=False,uniform=False)


                
        all_rets.append([sum(p['rewards']) for p in paths])




    if save_video:
        # save frames to file temporarily
        temp_dir = os.path.join(path_to_exp, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        for i, frm in enumerate(video_frames):
            frm.save(os.path.join(temp_dir, '%06d.jpg' % i))

        video_filename=os.path.join(path_to_exp, 'video.mp4'.format(idx))
        # run ffmpeg to make the video
        os.system('ffmpeg -i {}/%06d.jpg -vcodec mpeg4 {}'.format(temp_dir, video_filename))
        # delete the frames
        shutil.rmtree(temp_dir)

    # compute average returns across tasks
    n = min([len(a) for a in all_rets])
    rets = [a[:n] for a in all_rets]
    rets = np.mean(np.stack(rets), axis=0)
    for i, ret in enumerate(rets):
        print('trajectory {}, avg return: {} \n'.format(i, ret))


@click.command()
@click.argument('config', default=None)
@click.argument('path', default=None)
@click.option('--num_trajs', default=100)
@click.option('--deterministic', is_flag=True, default=False)
@click.option('--video', is_flag=True, default=False)
def main(config, path, num_trajs, deterministic, video):
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant, path, num_trajs, deterministic, video,sparse_reward=False,specific_context=False,import_z=False,action_error=True,always_uniform=False,
prior_untill_succ=False,high_prob_z=False,keep_succ_z=False,determinstic_z=False,sparse_reward_with_uncertain_information=False)


if __name__ == "__main__":
    main()
