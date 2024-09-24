import numpy as np

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer, SimpleReplayBuffer_enc
from gym.spaces import Box, Discrete, Tuple,Dict
from gym.spaces.utils import flatdim,flatten

class MultiTaskReplayBuffer(object):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            tasks,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param tasks: for multi-task setting
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.task_buffers = dict([(idx, SimpleReplayBuffer(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
        )) for idx in tasks])


    def add_sample(self, task, observation, action, reward, terminal,
            next_observation,target_offset, **kwargs):

        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        self.task_buffers[task].add_sample(
                observation, action, reward, terminal,
                next_observation,target_offset, **kwargs)

    def terminate_episode(self, task):
        self.task_buffers[task].terminate_episode()

    def random_batch(self, task, batch_size, sequence=False):
        if sequence:
            batch = self.task_buffers[task].random_sequence(batch_size)
        else:
            batch = self.task_buffers[task].random_batch(batch_size)
        return batch

    def num_steps_can_sample(self, task):
        return self.task_buffers[task].num_steps_can_sample()

    def add_path(self, task, path):
        self.task_buffers[task].add_path(path)

    def add_paths(self, task, paths):
        for path in paths:
            self.task_buffers[task].add_path(path)

    def clear_buffer(self, task):
        self.task_buffers[task].clear()


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)


    elif isinstance(space, Dict):
        return flatdim(space)


    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        # import OldBox here so it is not necessary to have rand_param_envs 
        # installed if not running the rand_param envs
        from rand_param_envs.gym.spaces.box import Box as OldBox
        if isinstance(space, OldBox):
            return space.low.size

        elif isinstance(space, np.ndarray):
            return np.prod(space.shape)
        else:
            raise TypeError("Unknown space: {}".format(space))






class MultiTaskReplayBuffer_enc(object):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            tasks,
            load_buffer=False,
            encoder=False
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param tasks: for multi-task setting
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.task_buffers = dict([(idx, SimpleReplayBuffer_enc(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),load_buffer=load_buffer,task=idx,encoder=encoder,
        )) for idx in tasks])

    
    def add_sample (self,o,a,no,r,task,itr_0=False,encoder=False,epoch=None,**kwargs):

        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        self.task_buffers[task].add_sample(o,a,no,r,task,itr_0=itr_0,encoder=encoder,epoch=epoch,**kwargs)


    
    def terminate_episode(self, task):
        self.task_buffers[task].terminate_episode()

    def random_batch(self, task, batch_size, sequence=False):
        if sequence:
            batch = self.task_buffers[task].random_sequence(batch_size)
        else:
            batch = self.task_buffers[task].random_batch(batch_size)
        return batch

    def num_steps_can_sample(self, task):
        return self.task_buffers[task].num_steps_can_sample()

    def add_path(self, task, path,itr_0=False,encoder=False,epoch=None):
        self.task_buffers[task].add_path(path,task,itr_0=itr_0,encoder=encoder,epoch=epoch)

    def add_paths(self, task, paths,itr_0=False,encoder=False,epoch=None):
        for path in paths:
            self.task_buffers[task].add_path(path['traj_context'],task,itr_0=itr_0,encoder=encoder,epoch=epoch)

    def clear_buffer(self, task):
        self.task_buffers[task].clear()


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)

    elif isinstance(space, Dict):
        return flatdim(space)


    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        # import OldBox here so it is not necessary to have rand_param_envs 
        # installed if not running the rand_param envs
        from rand_param_envs.gym.spaces.box import Box as OldBox
        if isinstance(space, OldBox):
            return space.low.size


        elif isinstance(space, np.ndarray):
            return np.prod(space.shape)
        else:
            raise TypeError("Unknown space: {}".format(space))
