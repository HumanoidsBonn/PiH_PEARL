import numpy as np
from gym import utils
from . import my_mujoco_env
from numpy import random

import os

def distance(a, b):
    assert a.shape == b.shape
    return np.sqrt(np.square(a[0]-b[0])+np.square(a[1]-b[1])+np.square(a[2]-b[2]))





class AbbEnv(my_mujoco_env.myMujocoEnv, utils.EzPickle):



    def __init__(self,distance_threshold=0.0005, max_stepsize=0.002,supposed_goal=[-0.2 ,0,0.79],max_offset_initial_pos=0.005): 
        utils.EzPickle.__init__(self)
        my_mujoco_env.myMujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "mymodel{}.xml".format(0)), 1,n_substeps=10,initial_qpos={'robot0:joint1':0 ,'robot0:joint4': 0,'robot0:joint6':0 ,'base:slide0':0,'base:slide1':0,'base:slide2':0,'robot0:joint2': 0.7858,'robot0:joint3': -0.4965,'robot0:joint5':1.2817},supposed_goal=supposed_goal ,max_offset_initial_pos=max_offset_initial_pos)

        self.max_stepsize=max_stepsize
        self.distance_threshold=distance_threshold
        self.supposed_goal= supposed_goal
        self.max_offset_initial_pos=max_offset_initial_pos
        print("init ABBENV")

    def _set_action(self, action,action_error=False):
        assert action.shape == (3,)
        action = action.copy()         
        actual_action=action* 0.002
        actual_action[2]=action[2]*0.002
        initial_pos= self.sim.data.get_body_xpos("tool").copy()
        self.move_linear (actual_action, initial_pos )
   

    def move_linear (self,actual_action, initial_pos ):
        minimum_step_size=0.00006
        max_normal_force=200
        prev_pos=initial_pos
        required_quat=[0.707,0,0.707,0]
        pos_ctrl=actual_action[0:3] 
        pos_ctrl[0]*=(((random.rand()*2)-1)/10)+1
        pos_ctrl[1]*=(((random.rand()*2)-1)/10)+1
        pos_ctrl[2]*=(((random.rand()*2)-1)/10)+1
        pos_ctrl=actual_action

        z_displacement=abs(pos_ctrl[2])

        if pos_ctrl[2] < 0 :
            move_down=True
            pos_ctrl[2]=0
            target_horizontal_pos = prev_pos + pos_ctrl
            self.sim.data.mocap_pos[:]=target_horizontal_pos
            self.sim.data.mocap_quat[:]=required_quat
            self.sim.forward()
            for _ in range(2):
                self.sim.step()
            self.sim.data.mocap_pos[:]=self.sim.data.get_body_xpos("tool").copy()
            self.sim.data.mocap_quat[:]=required_quat
            self.sim.forward()
            for _ in range(2):
                self.sim.step()
  
            first_ee_pos=self.sim.data.get_site_xpos("endeffector").copy()
            current_ee_pos=self.sim.data.get_site_xpos("endeffector").copy()
            supposed_next_ee_z_pos=current_ee_pos[2]-z_displacement
            if (supposed_next_ee_z_pos <0.80013) :
                if current_ee_pos[2]>0.80013:
                    self.sim.data.mocap_pos[0][2]=0.80013+0.09
                    self.sim.data.mocap_quat[:]=required_quat
                    self.sim.forward()
                    for _ in range(2):
                        self.sim.step()
                new_ee_pos=self.sim.data.get_site_xpos("endeffector").copy()
                remaining_z=new_ee_pos[2]-supposed_next_ee_z_pos
                number_of_steps=int(remaining_z/minimum_step_size)
                for _ in range (number_of_steps):
                    self.sim.data.mocap_pos[0][2]-=minimum_step_size
                    self.sim.data.mocap_quat[:]=required_quat
                    self.sim.forward()
                    for _ in range(2):
                        self.sim.step()
                    force_sensor=self.sim.data.sensordata.copy()
                    if force_sensor[0] > max_normal_force :
                        self.extra_force_flag=True
                        self.sim.data.mocap_pos[0][2]+=0.0005
                        self.sim.forward()
                        for _ in range(2):
                            self.sim.step()
                        break
                    new_ee_pos=self.sim.data.get_site_xpos("endeffector").copy()
                    if new_ee_pos[2]<0.7999:
                        self.sim.data.mocap_pos[0][2]=supposed_next_ee_z_pos+0.09
                        self.sim.forward()
                        for _ in range(2):
                            self.sim.step()
                        break
  
                        


            else:
                self.sim.data.mocap_pos[0][2]-=z_displacement
                self.sim.data.mocap_quat[:]=required_quat
                self.sim.forward()
                for _ in range(2):
                    self.sim.step()

        else:
            target_pos = prev_pos + pos_ctrl
            self.sim.data.mocap_pos[:]=target_pos
            self.sim.data.mocap_quat[:]=required_quat
            self.sim.forward()
            for _ in range(2):
                self.sim.step()
 
        current_pos=self.sim.data.get_body_xpos("tool").copy()

        counter=0
        while ( abs(prev_pos[0] - current_pos[0])> 0.002*1.2  or abs(prev_pos[1] - current_pos[1])> 0.002*1.2  or abs(prev_pos[2] - current_pos[2])> 0.002*1.2 ) and counter < 10 :
        
            new_proposed_action=  actual_action
            new_proposed_action[0]*= ((((random.rand()*2)-1)/10)+1)
            new_proposed_action[1]*= ((((random.rand()*2)-1)/10)+1)
            new_proposed_action[2]*= ((((random.rand()*2)-1)/10)+1) 
            new_proposed_goal = prev_pos + new_proposed_action
            complete_proposed_action=new_proposed_action.copy()
            z_displacement=abs(new_proposed_action[2])




            if new_proposed_action[2] < 0 :
                move_down=True

                new_proposed_action[2]=0
                target_horizontal_pos = prev_pos + new_proposed_action
                self.sim.data.mocap_pos[:]=target_horizontal_pos
                self.sim.data.mocap_quat[:]=required_quat
                self.sim.forward()
                for _ in range(2):
                    self.sim.step()
                self.sim.data.mocap_pos[:]=self.sim.data.get_body_xpos("tool").copy()
                self.sim.data.mocap_quat[:]=required_quat
                self.sim.forward()
                for _ in range(2):
                    self.sim.step()



                current_ee_pos=self.sim.data.get_site_xpos("endeffector").copy()
                supposed_next_ee_z_pos=current_ee_pos[2]-z_displacement
                if (supposed_next_ee_z_pos <0.80013) :
                    if current_ee_pos[2]>0.80013:
                        self.sim.data.mocap_pos[0][2]=0.80013+0.021
                        self.sim.data.mocap_quat[:]=required_quat
                        self.sim.forward()
                        for _ in range(2):
                            self.sim.step()
                    new_ee_pos=self.sim.data.get_site_xpos("endeffector").copy()
                    remaining_z=new_ee_pos[2]-supposed_next_ee_z_pos
                    number_of_steps=int(remaining_z/minimum_step_size)
                    for _ in range (number_of_steps):
                        self.sim.data.mocap_pos[0][2]-=minimum_step_size
                        self.sim.data.mocap_quat[:]=required_quat
                        self.sim.forward()
                        for _ in range(2):
                            self.sim.step()
                        force_sensor=self.sim.data.sensordata.copy()
                        if force_sensor[0] > max_normal_force :
                            self.extra_force_flag=True
                            self.next_obs_with_force= self._get_observation_with_force()
                            self.sim.data.mocap_pos[0][2]+=0.0005
                            self.sim.forward()
                            for _ in range(2):
                                self.sim.step()
                            break
                        current_ee_pos=self.sim.data.get_site_xpos("endeffector").copy()
                        if current_ee_pos[2]<0.7998:
                            self.sim.data.mocap_pos[0][2]=new_proposed_goal[2]
                            self.sim.forward()
                            for _ in range(2):
                                self.sim.step()
                            break
  

                else:
                    self.sim.data.mocap_pos[0][2]-=z_displacement
                    self.sim.data.mocap_quat[:]=required_quat
                    self.sim.forward()
                    for _ in range(2):
                        self.sim.step()



            else:
                target_pos = prev_pos + new_proposed_action
                self.sim.data.mocap_pos[:]=target_pos
                self.sim.data.mocap_quat[:]=required_quat
                self.sim.forward()
                for _ in range(2):
                    self.sim.step()

      
            current_pos=self.sim.data.get_body_xpos("tool").copy()
           
         


            counter+=1

        

    def reset(self,randomize_initial_pos=True):

        
        self._env_setup(self.initial_qpos,randomize_initial_pos=randomize_initial_pos)
        self.goal = self.sim.data.get_site_xpos("target" ).copy()
        
       
        obs = self._get_observation()
        
        return obs


