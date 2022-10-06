import numpy as np
import pandas as pd
import gym
from gym import spaces

#influent generation
from gym_SBR.envs import buffer_tank3 as buffer_tank
from gym_SBR.envs import SBR_model_continuous as SBR

from gym_SBR.envs.module_reward_continuous import sbr_reward
from gym_SBR.envs.module_temperature import DO_set
from gym_SBR.envs.module_batch_time import batch_time
from gym_SBR.envs.component_figure import components

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

global WV, IV, Qin, t_ratio, t_cycle

# Plant Config.
WV = 1.32  # m^3, Working Volume

t_ratio = [4.2/100, 8.3/100, 37.5/100, 31.2/100, 2.1/100, 8.3/100, 2.1/100, 6.3/100]
t_delta = 0.002  /24

# phase time
t_cycle = 12 / 24  # hour -> day, 12hr

t_memory1, t_memory2, t_memory3, t_memory4, t_memory5, t_memory6,t_memory7, t_memory8= batch_time(t_cycle, t_ratio, t_delta)


#Oxygen concentration at saturation : 15deg
So_sat = DO_set(15)

# DO control prarameters
DO_control_par = [5.0, 0.00035, 0.02/24, 2, 0, 240, 12, 2, 5, 0.005, So_sat]
# Kc, taui, delt, So_set, Kla_min, Kla_max, DKla_max, So_low, So_high, DO saturation

dt = DO_control_par[2]

#DO control setpoints
DO_setpoints = [0,0,2,0,2,0,0,2]

kla0 = 0

class SbrEnv3(gym.Env):
    """custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    
    def __init__(self):
        self.action_space = spaces.Box(np.array([0.,0.,0.]), np.array([1.0,1.0,1.0]), dtype=np.float32)
                                       
        self.observation_space= spaces.Box(low=np.array([0.5,0,0]), high= np.array([1.33,2.5,2]), dtype = np.float32 )
        self.reward = 0
        
    
    def reset(self):
        global influent_mixed
        global WV, IV
        global x0, x0_init
        global IV_init, Qin
        global state

        # initial state from stablization
        x0_init = [0.6161484733495801, 30, 0.571098000538576, 1440.01157895393,
                   31.254221999137, 2599.2714348941, 168.915006750837, 551.901552960823, 2.16607843793004,
                   13.3791460027604, 0.00562880208518134, 0.35996687629947, 1.86916737961228, 3.790463057094611]


        # initial Inoculum Volume

        IV_init = 0.6161484733495801  # m^3, Inoculum Volume

        # Volume update from "Step"
        if 'IV_new' in globals():
            IV = IV_init#IV_new
        if 'IV_new' not in globals():
            IV = IV_init

        Qin = WV - IV

        # Initial states from "Step"
        if 'x0_new' in globals():
            x0 = x0_init#x0_new
        if 'x0_new' not in globals():
            x0 = x0_init
            
        print('X0: {}'.format(x0))

        # Load: generated influent
        switch, influent_mixed, influent_var = buffer_tank.influent.buffer_tank(0)#np.random.choice(8,1))
  


        state_instant1 = np.append([x0],[influent_mixed], axis=0)  # 한번 시도
        state_instant2 = np.sum(state_instant1, axis=0)
        
        Vv_in = state_instant2[0]
        COD_in1 = state_instant2[1]+state_instant2[2]+state_instant2[3]+state_instant2[4]+state_instant2[5]+state_instant2[6]+state_instant2[7]
        Snh_in1 = state_instant2[10]
        
        COD_in2 = (COD_in1-5145)/10
        Snh_in2 = (Snh_in1)/30

        state = np.array([Vv_in,COD_in2, Snh_in2])
        print("State in reset: {}".format(state))
       
       
        
        return state

    def _next_observation(self, WV, IV, t_ratio, influent_mixed, DO_control_par, x0, DO_setpoints,kla0):
        t, x, x_last, sp_memory3, So_memory3,t_memory3, sp_memory5, So_memory5, t_save5, sp_memory8, So_memory8, t_save8,Qeff, eff,Qw,kla3,kla5, kla8,EQI = SBR.run(WV, IV, t_ratio, influent_mixed, DO_control_par,x0, DO_setpoints, kla0)


        return  t, x, x_last, sp_memory3, So_memory3,t_memory3, sp_memory5, So_memory5, t_save5, sp_memory8, So_memory8, t_save8,Qeff, eff,Qw,kla3,kla5, kla8,EQI
    
    def step(self, action) :
        
        action = np.clip(action, self.action_space.low, self.action_space.high)


        global influent_mixed
        global x_last,x, x0,x0_new,x0_init
        global WV,IV_new

        #Execute one time steo within the environment
        self._take_action(action)

        # Filling phase동안 들어오는 유입수 유량
        influent_mixed[0] = Qin/(t_cycle*t_ratio[0])

        
        print("DO setpoint after take_action in step: {}".format(DO_setpoints))


        # SBR 돌리기
        t, x, x_last, sp_memory3, So_memory3,t_memory3, sp_memory5, So_memory5, t_save5, sp_memory8, So_memory8, t_save8,Qeff, eff,Qw,kla3,kla5, kla8,EQI= self._next_observation(WV, IV, t_ratio, influent_mixed, DO_control_par, x0, DO_setpoints, kla0)
        x0_new = x_last
        IV_new = x_last[0]
        
        Snh = eff[3]
        reward, OCI = sbr_reward(DO_control_par, kla3, kla5, kla8,Qw, EQI,Qin, Qeff, Snh,DO_setpoints)
        
        print("REWARD: {}".format(reward))

        self.reward = reward

        done = True
        
        
        COD_eff = eff[2]
        Snh_eff = eff[3]/30
        
        state = np.array([Qeff,COD_eff, Snh_eff])

      
        return  state, reward, done, {}
    
    def _take_action(self, action):
        
        #global global_rewards, global_episodes
        global t_memory1, t_memory2, t_memory3, t_memory4, t_memory5, t_memory8
        global So_memory1, So_memory2, So_memory3, So_memory4, So_memory5, So_memory8
        global sp_memory1, sp_memory2, sp_memory3, sp_memory4, sp_memory5, sp_memory8
        global t, soln
        

       
        
        DO_setpoints[2] = action[0]*8
        DO_setpoints[4] = action[1]*8
        DO_setpoints[7] = action[2]*8


    def render(self, mode='human', close=False): 
        
        #print("Episode {}".format(global_episodes))
        print("Reward for this episode: {}".format(self.reward))
        #print("action for this episode: {}".format(action))
