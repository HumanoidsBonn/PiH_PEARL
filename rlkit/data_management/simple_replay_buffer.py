import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer,ReplayBuffer_enc


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._sparse_rewards = np.zeros((max_replay_buffer_size, 1))


        self._target_offsets=np.zeros((max_replay_buffer_size, 3))
        self._actual_prev_states=np.zeros((max_replay_buffer_size, 3))
        self._actual_next_states=np.zeros((max_replay_buffer_size, 3))
  
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self.clear()

    def add_sample(self, observation, action, reward, terminal,
                   next_observation,target_offset,actual_prev_state,actual_next_state, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._target_offsets[self._top] = target_offset
        self._actual_prev_states[self._top]=actual_prev_state
        self._actual_next_states[self._top]=actual_next_state
        self._sparse_rewards[self._top] = kwargs['env_info'].get('sparse_reward', 0)
        self._advance()

    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self._top

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            target_offsets=self._target_offsets[indices],
            next_observations=self._next_obs[indices],
            actual_prev_states=self._actual_prev_states[indices],
            actual_next_states=self._actual_next_states[indices],
            sparse_rewards=self._sparse_rewards[indices],
        )

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            # TODO hack to not deal with wrapping episodes, just don't take the last one
            start = np.random.choice(self.episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def num_steps_can_sample(self):
        return self._size






class SimpleReplayBuffer_enc(ReplayBuffer_enc):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,load_buffer=False,task=None,encoder=False,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._o = np.zeros((max_replay_buffer_size, observation_dim))
        self._a = np.zeros((max_replay_buffer_size, observation_dim))
        self._r = np.zeros((max_replay_buffer_size, 1))
        self._no = np.zeros((max_replay_buffer_size, observation_dim))
        self._task=task        
        self.clear()
    

    def add_sample(self,o,a,no,r,task,itr_0=False,encoder=False,epoch=None,**kwargs):


        self._o[self._top] = o
        self._a[self._top] = a
        self._r[self._top] = r
        self._no[self._top] = no
        self._advance()


    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self._top

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        return dict(

            o=self._o[indices], 
            a=self._a[indices], 
            r=self._r[indices] ,
            no=self._no[indices] ,
       
        )

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            # TODO hack to not deal with wrapping episodes, just don't take the last one
            start = np.random.choice(self.episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def num_steps_can_sample(self):
        return self._size














