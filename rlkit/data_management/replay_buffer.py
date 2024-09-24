import abc


class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal,target_offset, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """

        for i, (
                obs,
                action,
                reward,
                next_obs,
                target_offset,
                terminal,
                agent_info,
                env_info, 
                actual_prev_state,
                actual_next_state,
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["target_offsets"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
            path["actual_prev_states"],
            path["actual_next_states"],
        )):
            self.add_sample(
                obs,
                action,
                reward,
                terminal,
                next_obs,
                target_offset,
                actual_prev_state,
                actual_next_state,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass














class ReplayBuffer_enc(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self,o,a,no,r,task,itr_0=False,encoder=False,epoch=None,**kwargs):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path,task,itr_0=False,encoder=False,epoch=None):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        '''
        print("observations == ",path["observations"].shape)
        print("actions == ",path["actions"].shape)
        print("rewards == ",path["rewards"].shape)
        print("next_observations == ",path["next_observations"].shape)
        print("target_offsets == ",path["target_offsets"].shape)
        print("agent_infos == ",len(path["agent_infos"]))
        print("agent_infos sample == ",path["agent_infos"][0])
        print("env_infos == ",len(path["env_infos"]))
        print("actual_prev_states == ",path["actual_prev_states"].shape)
        print("actual_next_states == ",path["actual_next_states"].shape)
        print("actual_motions_towards_goal_x == ",path["actual_motions_towards_goal_x"].shape)
        print("actual_motions_towards_goal_y == ",path["actual_motions_towards_goal_y"].shape)
        print("actual_motions_towards_goal_z == ",path["actual_motions_towards_goal_z"].shape)
        print("actual_rotations_towards_goal == ",path["actual_rotations_towards_goal"].shape)
        '''
        for i in path:

            self.add_sample(i[0],i[1],i[2],i[3],task,itr_0=itr_0,encoder=encoder,epoch=epoch)

        self.terminate_episode()

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass



