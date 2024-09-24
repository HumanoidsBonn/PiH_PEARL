import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch

def rollout(env, agent, max_path_length=np.inf, accum_context=True, animated=False, save_frames=False,scale_std=None, sparse_reward=False,specific_context=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """


    traj_context=[]
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []

    target_offsets=[]
    actual_prev_states=[]
    actual_next_states=[]

    agent_info=None
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    counter=0


    sparse_reward=sparse_reward
    successful_traj=False

    while path_length < max_path_length:
        policy_outputs=None
        counter=counter+1
        if animated:
            policy_outputs=agent.get_action(o,return_parameters=True,stricted_std=False,scale_std=scale_std)
            a=policy_outputs[0].squeeze().to('cpu').detach().numpy()



        else:
            a, agent_info = agent.get_action(o)


        
        next_o, r, d, env_info = env.step(a,sparse_reward=sparse_reward)
      
        target_offset=env_info['target_offset']
        actual_prev_state=env_info['actual_prev_state']
        actual_next_state=env_info['actual_next_state']

        total_motion_towards_goal=env_info['total_motion_towards_goal']

        if sparse_reward:
            actual_prev_state[0:-1]=[0,0]
            actual_next_state[0:-1]=[0,0]
            traj_context.append([o, a, next_o,total_motion_towards_goal]) 
        else:
            traj_context.append([o, a, next_o,total_motion_towards_goal])

        if env_info['is_success'] !=0:
            successful_traj=True

        if animated:
            env.render()

            
                                  
            print("************************************") 
            print("************************************") 

        
            if counter ==0:
                print("        ")
                print(" ************* new trajectory***********")
                print("        ")
        
               
            print(" target offset in mm==",target_offset) 

            print("Z mean == ",agent.z_means)
            print("Z std== ",torch.sqrt(agent.z_vars.detach().to(device=torch.device("cpu"))))
            print("Z == ",agent.z)

          
            
            print("actual motion in X == ",env_info['actual_motion_towards_goal_x']/2)
            print("actual motion in Y == ",env_info['actual_motion_towards_goal_y']/2)
            print("actual motion in Z== ",env_info['actual_motion_towards_goal_z']/2)

            print("action by policy ==",policy_outputs[0])

            print("action mean ==",policy_outputs[1])

            print("action std== ",policy_outputs[5])
            print("reward== ",r)
            print("X-dist == ", (env.sim.data.get_site_xpos("endeffector").copy()[0]-env.sim.data.get_site_xpos("target").copy()[0])*1000)
            print("Y-dist == ", (env.sim.data.get_site_xpos("endeffector").copy()[1]-env.sim.data.get_site_xpos("target").copy()[1])*1000)


        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        target_offsets.append(target_offset)
        actual_prev_states.append(actual_prev_state)
        actual_next_states.append(actual_prev_state)
        path_length += 1
        o = next_o

        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d: ## done
            break



        # update the agent's current context
    if specific_context:
        if accum_context and env_info['is_success']:
            for i in traj_context:
                agent.update_context(i)

    else:
        if accum_context:
            for i in traj_context:
                agent.update_context(i)



    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)

    target_offsets=np.array(target_offsets)
    if len(target_offsets.shape) == 1:
        target_offsets = np.expand_dims(target_offsets, 1)
    observations = np.array(observations)


    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )


    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        successful_traj=successful_traj,
        target_offsets=target_offsets,
        actual_prev_states=actual_prev_states,
        actual_next_states=actual_next_states,
        traj_context=traj_context,
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
