#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import gym
from gym import spaces

#influent generation
from gym_SBR.envs import buffer_tank2 as buffer_tank
from gym_SBR.envs import SBR_model_batchPID_fbPID as SBR
from gym_SBR.envs import SBR_model_PID_off as SBR_PID_off
from gym_SBR.envs import SBR_model_PID_on as SBR_PID_on

from gym_SBR.envs.module_reward import sbr_reward
from gym_SBR.envs.module_batch_PID import batch_PID
from gym_SBR.envs.module_temperature import DO_set
from gym_SBR.envs.module_batch_time import batch_time

# create a list for string global rewards and episodes




global_rewards = []
global_episodes = 0

t_history = []
DO_history = []
output_history = []
reward_history = []
action_history = []

kla3_history = []
kla5_history = []
kla8_history = []

# Plant Config.
WV = 1.32  # m^3, Working Volume
IV = 0.66  # m^3, Inoculum Volume

t_ratio = [4.2/100, 8.3/100, 37.5/100, 31.2/100, 2.1/100, 8.3/100, 2.1/100, 6.3/100]
t_delta = 0.002  /24

# phase time
t_cycle = 12 / 24  # hour -> day, 12hr

t_memory1, t_memory2, t_memory3, t_memory4, t_memory5, t_memory6, t_memory7, t_memory8 = batch_time(t_cycle, t_ratio, t_delta)


# Memory

memory_switch= []
memory_influent_mixed = []
memory_influent_var= []

memory_component_state = []
memory_time = []
memory_e_batch_1 = np.zeros((1, len(t_memory1)))
memory_e_batch_2 = np.zeros((1, len(t_memory2)))
memory_e_batch_3 = np.zeros((1, len(t_memory3)))
memory_e_batch_4 = np.zeros((1, len(t_memory4)))
memory_e_batch_5 = np.zeros((1, len(t_memory5)))
memory_e_batch_8 = np.zeros((1, len(t_memory8)))

u_batch_1 = np.zeros((1, len(t_memory1)))
u_batch_2 = np.zeros((1, len(t_memory2)))
u_batch_3= np.zeros((1, len(t_memory3)))
u_batch_4 = np.zeros((1, len(t_memory4)))
u_batch_5 = np.zeros((1, len(t_memory5)))
u_batch_8 = np.zeros((1, len(t_memory8)))

# initial state from stablization
x0 = [IV, 30.0, 0.5601630529230822, 1762.3890076468106, 30.97046860269441, 2628.6551849696393, 188.71238190722482,
      780.479571994941, 6.83620016588177, 14.575400491942467, 0.00872090237410032, 0.36940333660700486,
      1.896711744868243, 3.705237172170034]

# Load: generated influent
switch, influent_mixed, influent_var = buffer_tank.influent.buffer_tank(0,12)

influent_mixed[0] = 31.4285 # 단위 변환

memory_switch.append(switch)
memory_influent_mixed.append(influent_mixed)
memory_influent_var.append(influent_var)

#Oxygen concentration at saturation : 15deg
So_sat = DO_set(15)

# DO control prarameters
DO_control_par = [0.5/1.18, 0.0015, 0.05, 2, 0, 240, 12, 2, 5, 0.005, So_sat]
# Kc, taui, delt, So_set, Kla_min, Kla_max, DKla_max, So_low, So_high, DO saturation

# Batch PID
par_batchPID =  [0.002018, 0.003643, 0.004036, 0, 0.01875, 0.0004671, 0.01564, 0.003643, 0.001028, 0,0,0,0,0, 0.003027, 0.003643] #tau_w, theta_w\


dt = DO_control_par[2]
#DO control setpoints
DO_setpoints = [0,0,2,0,2,0,0,2]




# Run SBR model at cycle 0, which is the non-control state

t, x, x_last, kla_memory1, So_memory1, sp_memory1, t_save1, kla_memory2, So_memory2, sp_memory2, t_save2, kla_memory3, So_memory3, sp_memory3, t_save3, kla_memory4, So_memory4, sp_memory4, t_save4, kla_memory5, So_memory5, sp_memory5, t_save5, kla_memory8, So_memory8, sp_memory8, t_save8 = \
    SBR_PID_on.run(WV, IV, t_ratio, influent_mixed, DO_control_par,x0, DO_setpoints)


sp_memory_1 = sp_memory1
So_memory_1 = So_memory1
sp_memory_2 = sp_memory2
So_memory_2 = So_memory2
sp_memory_3 = sp_memory3
So_memory_3 = So_memory3
sp_memory_4 = sp_memory4
So_memory_4 = So_memory4
sp_memory_5 = sp_memory5
So_memory_5 = So_memory5
sp_memory_8 = sp_memory8
So_memory_8 = So_memory8

kla_memory_1 = kla_memory1
kla_memory_2 = kla_memory2
kla_memory_3 = kla_memory3
kla_memory_4 = kla_memory4
kla_memory_5 = kla_memory5
kla_memory_8 = kla_memory8

kla_memory_1_1 = kla_memory1
kla_memory_2_1 = kla_memory2
kla_memory_3_1 = kla_memory3
kla_memory_4_1 = kla_memory4
kla_memory_5_1 = kla_memory5
kla_memory_8_1 = kla_memory8



class SbrEnv(gym.Env):
    """custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    
    def __init__(self):
        self.action_space = spaces.Box(np.array([0.0,0.0,0.0]), np.array([5.0,5.0,5.0]), dtype=np.float32)
                                       #np.array([0,0,0]), high = np.array([5,5,5]), dtype=np.float16)
        self.observation_space= spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0]), high= np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2]), dtype = np.float32 )
        self.reward = 0
        
    
    def reset(self):
            #influent_mixed[0] = 0.66 # 단위 변환
            
            #state = x_last        
            state_instant = np.append([x_last],[influent_mixed], axis=0)  # 한번 시도
            state = np.sum(state_instant, axis=0)
            
            state[0] = 1#state[0]/1.32
            state[1] = state[1]/60
            state[2] = state[2]/31
            state[3] = state[3]/1974
            state[4] = state[4]/107
            state[5] = state[5]/2237
            state[6] = state[6]/195
            state[7] = state[7]/988
            state[8] = state[8]/2
            state[9] = state[9]/4
            state[10] = state[10]/14
            state[11] = state[11]/3
            state[12] = state[12]/5
            state[13] = state[13]/12
                        
    
            return state

    def _next_observation(self, WV, IV, t_ratio, influent_mixed, DO_control_par, x_last, DO_setpoints, u_batch_1,  u_batch_2,   u_batch_3, u_batch_4, u_batch_5, u_batch_8, kla_memory_1_1, kla_memory_2_1, kla_memory_3_1, kla_memory_4_1, kla_memory_5_1, kla_memory_8_1):
        
        t,soln, x_last, t_memory1, sp_memory1, So_memory1, t_memory2, sp_memory2, So_memory2, t_memory3, sp_memory3, So_memory3, t_memory4, sp_memory4, So_memory4, t_memory5, sp_memory5, So_memory5, t_memory8, sp_memory8, So_memory8, kla_memory1, kla_memory2, kla_memory3, kla_memory4, kla_memory5, kla_memory8, Qeff,Qw = SBR.run(WV, IV, t_ratio, influent_mixed, DO_control_par, x_last, DO_setpoints, u_batch_1[-1, :], u_batch_2[-1, :], u_batch_3[-1, :], u_batch_4[-1, :], u_batch_5[-1, :], u_batch_8[-1, :], kla_memory_1_1,  kla_memory_2_1, kla_memory_3_1, kla_memory_4_1, kla_memory_5_1, kla_memory_8_1)
        
        
        return  t,soln, x_last, t_memory1, sp_memory1, So_memory1, t_memory2, sp_memory2, So_memory2, t_memory3, sp_memory3, So_memory3, t_memory4, sp_memory4, So_memory4, t_memory5, sp_memory5, So_memory5,   t_memory8, sp_memory8, So_memory8,    kla_memory1, kla_memory2, kla_memory3, kla_memory4, kla_memory5, kla_memory8,   Qeff,Qw
    
    def step(self, action) :
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        global influent_mixed
        global x_last   
        influent_mixed[0] = 31.4285 # 단위 변환
        
        print("____action(clipped) in Gym: {}".format(action))


        #Execute one time steo within the environment
        self._take_action(action)
    
    
        t,soln, x_last, t_memory1, sp_memory1, So_memory1, t_memory2, sp_memory2, So_memory2, t_memory3, sp_memory3, So_memory3, t_memory4, sp_memory4, So_memory4, t_memory5, sp_memory5, So_memory5, t_memory8, sp_memory8, So_memory8, kla_memory1, kla_memory2, kla_memory3, kla_memory4, kla_memory5, kla_memory8, Qeff,Qw = self._next_observation(WV, IV, t_ratio, influent_mixed, DO_control_par, x_last, DO_setpoints, u_batch_1,  u_batch_2,   u_batch_3, u_batch_4, u_batch_5, u_batch_8, kla_memory_1_1, kla_memory_2_1, kla_memory_3_1, kla_memory_4_1, kla_memory_5_1, kla_memory_8_1)
                
        
        reward =  sbr_reward(x_last,  DO_control_par, kla_memory_3, kla_memory_5, kla_memory_8, Qeff,Qw)
        self.reward = reward

        done = True

        switch, influent_mixed, influent_var = buffer_tank.influent.buffer_tank(0,12)

        x = x_last
        
        #state = x_last
        state_instant = np.append([x],[influent_mixed], axis=0)  # 한번 시도
        state = np.sum(state_instant, axis=0)  
        
        
        state[0] = 1#state[0]/1.32
        state[1] = state[1]/60
        state[2] = state[2]/31
        state[3] = state[3]/1974
        state[4] = state[4]/107
        state[5] = state[5]/2237
        state[6] = state[6]/195
        state[7] = state[7]/988
        state[8] = state[8]/2
        state[9] = state[9]/4
        state[10] = state[10]/14
        state[11] = state[11]/3
        state[12] = state[12]/5
        state[13] = state[13]/12
        
        
        
        return  state, reward, done, {}
    
    def _take_action(self, action):
        
        global global_rewards, global_episodes
        global t_memory1, t_memory2, t_memory3, t_memory4, t_memory5, t_memory8
        global So_memory1, So_memory2, So_memory3, So_memory4, So_memory5, So_memory8
        global sp_memory1, sp_memory2, sp_memory3, sp_memory4, sp_memory5, sp_memory8
        global memory_e_batch_1, memory_e_batch_2, memory_e_batch_3, memory_e_batch_4, memory_e_batch_5, memory_e_batch_8
        global u_batch_1, u_batch_2, u_batch_3, u_batch_4, u_batch_5, u_batch_8
        global t, soln
       
                  
        
        DO_setpoints[2] = action[0]
        DO_setpoints[4] = action[1]
        DO_setpoints[7] = action[2]
        
        sp_memory3_1 = sp_memory3[:]/sp_memory3[0]*action[0]
        sp_memory5_1 = sp_memory5[:] / sp_memory5[0] *action[1]
        sp_memory8_1 = sp_memory8[:] / sp_memory8[0] *action[2]
        
        
        u_batch_1, u_batch_2, u_batch_3, u_batch_4, u_batch_5, u_batch_8, memory_e_batch_1, memory_e_batch_2, memory_e_batch_3, memory_e_batch_4, memory_e_batch_5, memory_e_batch_8 = batch_PID(par_batchPID, t_memory1, t_memory2, t_memory3, t_memory4, t_memory5, t_memory8, t_delta, So_memory1, So_memory2, So_memory3, So_memory4, So_memory5, So_memory8,  sp_memory1, sp_memory2, sp_memory3_1, sp_memory4, sp_memory5_1, sp_memory8_1, memory_e_batch_1, memory_e_batch_2, memory_e_batch_3, memory_e_batch_4, memory_e_batch_5, memory_e_batch_8, u_batch_1, u_batch_2, u_batch_3, u_batch_4, u_batch_5, u_batch_8)



    def render(self, mode='human', close=False): 
        
        #print("Episode {}".format(global_episodes))
        print("Reward for this episode: {}".format(self.reward))
        #print("action for this episode: {}".format(action))
