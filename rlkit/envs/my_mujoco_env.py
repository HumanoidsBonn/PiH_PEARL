
from collections import OrderedDict
import os
from numpy import random

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

from rlkit.core.serializable import Serializable


try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class myMujocoEnv(gym.Env,Serializable):
    """just for Abb model.
    """

    def __init__(self, model_path, frame_skip,n_substeps=20,initial_qpos=None,supposed_goal=[-0.2,0,0.79],max_offset_initial_pos=0.005):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", "mymodel0.xml")
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model,nsubsteps=n_substeps)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self.n_substeps=n_substeps  #number of simulation steps per step
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.initial_qpos = initial_qpos or self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self._env_setup(initial_qpos=self.initial_qpos)
        self.goal = self.sim.data.get_site_xpos("target" ).copy()
        self._goal=self.sim.data.get_site_xpos("target" ).copy()
        self.supposed_goal=supposed_goal
        self.initial_state=self.sim.get_state()
        obs = self._get_observation()
        self.observation_space=obs

        self.action_space=np.zeros((3))
        self.seed()

        self.max_offset_initial_pos=max_offset_initial_pos

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError
    def viewer_setup(self):
        body_id = self.sim.model.body_name2id('tool')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    # -----------------------------



    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip*self.n_substeps

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def _env_setup(self, initial_qpos,randomize_initial_pos=True):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)

        self.sim.forward()
        self.sim.data.mocap_pos[:]=self.sim.data.body_xpos[11].copy()

        self.sim.data.mocap_quat[:]=[0.707,0,0.707,0]

        self.sim.forward()
        for _ in range(2):
            self.sim.step()
        original_ee_pos=self.sim.data.get_site_xpos("endeffector")

        
        if randomize_initial_pos:
            offset=((random.rand(3)*2)-1)*0.002
            self.sim.data.mocap_pos[:]+=offset
            self.sim.forward()
            for _ in range(2):
                self.sim.step()

    def _get_obs(self):
        endeffector_pos = self.sim.data.get_site_xpos("endeffector").copy()
        forcesensor=self.sim.data.sensordata

        return {
            'endeffector_pos': endeffector_pos.copy(),
            'forcesensor': forcesensor.copy(),
            'desired_goal': self.goal.copy(),
        }


    def _get_observation(self):
        endeffector_pos = self.sim.data.get_site_xpos("endeffector")
        forcesensor=self.sim.data.sensordata       
        forcesensor=np.clip(forcesensor,-100,100)
        goal=np.array(self.supposed_goal)
        normalized_z= (((goal[2]-endeffector_pos[2])*1000)/20)*2
        normalized_y= (((goal[1]-endeffector_pos[1])*1000)/5)*2
        normalized_x= (((goal[0]-endeffector_pos[0])*1000)/5)*2
        
        return np.array([normalized_x,normalized_y,normalized_z])
     


def distance(a, b):
    return np.linalg.norm(a - b, axis=-1)




    


    def init_serialization(self, locals):
        Serializable.quick_init(self, locals)

    def log_diagnostics(self, *args, **kwargs):
        pass




