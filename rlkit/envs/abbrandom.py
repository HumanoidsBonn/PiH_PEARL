import numpy as np

from .abb import AbbEnv
from . import register_env

from . import my_mujoco_env

import os

def distance(a, b):
    return np.linalg.norm(a - b, axis=-1)




@register_env('abb-random')
class AbbRandom (AbbEnv):

    
    def __init__(self, task={ },n_tasks=8,supposed_goal=[-0.2,0,0.79],
randomize_tasks=False,max_offset_initial_pos=0.005,randomize_initial_pos=True,max_stepsize=0.002,**kwargs):        

        super(AbbRandom, self).__init__(distance_threshold=0.001, max_stepsize=max_stepsize,supposed_goal=supposed_goal,max_offset_initial_pos=max_offset_initial_pos)

        self.tasks = self.sample_tasks(n_tasks)
        self._task = task

        self._goal=self.goal
        self.supposed_goal=supposed_goal
   
        self.randomize_initial_pos=randomize_initial_pos

        self.max_stepsize=max_stepsize
        self.max_offset_initial_pos=max_offset_initial_pos

    def get_all_task_idx(self):
        return range(len(self.tasks))


    def reset_task(self, idx):
        self._task = self.tasks[idx]
        my_mujoco_env.myMujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "mymodel{}.xml".format(idx)) , 1,n_substeps=10,initial_qpos={'robot0:joint1':0 ,'robot0:joint4': 0,'robot0:joint6':0 ,'base:slide0':0,'base:slide1':0,'base:slide2':0,'robot0:joint2': 0.7858,'robot0:joint3': -0.4965,'robot0:joint5':1.2817},supposed_goal=self.supposed_goal )



        self.reset(randomize_initial_pos=True)

    def step(self, action,sparse_reward=False,action_error=False):
        act=action
        prev_obs=self._get_observation().copy()
        prev_end_effector=self.sim.data.get_site_xpos("endeffector").copy()
        self._set_action(action,action_error=action_error)

        for _ in range (2):	
            self.sim.step()
        next_obs = self._get_observation().copy()

        next_end_effector=self.sim.data.get_site_xpos("endeffector").copy()
        done = False
        info = {'is_success': self._is_success(next_obs[0:3], self.sim.data.get_body_xpos("target" ).copy()),
        }
        
        info['in_hole']=False


        target_offset_mm=(self.sim.data.get_site_xpos("target").copy()-np.array(self.supposed_goal))*1000
    
        info['target_offset']=target_offset_mm.flatten()

        actual_goal=self.sim.data.get_site_xpos("target").copy()
        actual_prev_state_x=(((actual_goal[0]-prev_end_effector[0])*1000)/5)*2
        actual_prev_state_y=(((actual_goal[1]-prev_end_effector[1])*1000)/5)*2
        actual_prev_state_z=(((actual_goal[2]-prev_end_effector[2])*1000)/20)*2
        actual_prev_state=np.array([ actual_prev_state_x, actual_prev_state_y, actual_prev_state_z])
        actual_prev_state=actual_prev_state.flatten()

        actual_next_state_x=(((actual_goal[0]-next_end_effector[0])*1000)/5)*2
        actual_next_state_y=(((actual_goal[1]-next_end_effector[1])*1000)/5)*2
        actual_next_state_z=(((actual_goal[2]-next_end_effector[2])*1000)/20)*2
        actual_next_state=np.array([actual_next_state_x,actual_next_state_y,actual_next_state_z])
        actual_next_state=actual_next_state.flatten()


        actual_motion_towards_goal_x = abs(((actual_goal[0]-prev_end_effector[0])*1000))-abs(((actual_goal[0]-next_end_effector[0])*1000))
        actual_motion_towards_goal_y = abs(((actual_goal[1]-prev_end_effector[1])*1000))-abs(((actual_goal[1]-next_end_effector[1])*1000))
        actual_motion_towards_goal_z = abs(((actual_goal[2]-prev_end_effector[2])*1000))-abs(((actual_goal[2]-next_end_effector[2])*1000))

        info['actual_motion_towards_goal_x']=actual_motion_towards_goal_x #*error_x
        info['actual_motion_towards_goal_y']=actual_motion_towards_goal_y #*error_y
        info['actual_motion_towards_goal_z']=actual_motion_towards_goal_z #*(0.5+error_z)

        info['total_motion_towards_goal']= (actual_motion_towards_goal_x+actual_motion_towards_goal_y+actual_motion_towards_goal_z)/3
        info['actual_prev_state']=actual_prev_state
        info['actual_next_state']=actual_next_state

        if self.sim.data.get_site_xpos("endeffector").copy()[2]-self.sim.data.get_body_xpos("target" ).copy()[2] < 0.009 :
            info ['in_hole']=True  
        
        if sparse_reward:
            reward=info['is_success']
        else:
            reward = self.compute_reward(prev_obs=prev_obs,next_obs=next_obs,prev_end_effector=prev_end_effector,next_end_effector=next_end_effector,action=action)

        return next_obs, reward, done, info


    def _is_success(self, achieved_goal, desired_goal):
        end_effector_pos=self.sim.data.get_site_xpos("endeffector").copy()
        actualgoal=self.sim.data.get_site_xpos("target").copy()
        sparse_reward=0.0
        if end_effector_pos[2]-actualgoal[2]< 0.009:
            sparse_reward=1.0
        else:
            sparse_reward=0
        
        return sparse_reward



    def compute_reward(self, prev_obs=None,next_obs=None,info=None,next_end_effector=None,prev_end_effector=None,action=None):
 
        endeffector_pos = self.sim.data.get_site_xpos("endeffector").flat
        actualgoal=self.sim.data.get_site_xpos("target" ).copy()
        distance= (( (endeffector_pos[0]-actualgoal[0])*1000)**2) + ( ((endeffector_pos[1]-actualgoal[1])*1000)**2) + ( ((endeffector_pos[2]-actualgoal[2])*500)**2) 
        reward= (distance/100) 
        return   -reward 


    def sample_tasks(self, num_tasks):
        tasks = [i  for i in range(num_tasks)]
        return tasks











