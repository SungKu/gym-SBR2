# Call:  Libraries
import numpy as np
import gym
from gym import spaces
import scipy.integrate as integrate
import copy
import math
import matplotlib.pyplot as plt

# Call: Modules
from gym_SBR.envs import buffer_tank3 as buffer_tank
from gym_SBR.envs import SBR_model_continuous as SBR
from gym_SBR.envs.module_reward_continuous1 import sbr_reward
from gym_SBR.envs.module_temperature import DO_set
from gym_SBR.envs.module_batch_time import batch_time
from gym_SBR.envs.component_figure import components

global WV, IV, Qin, t_ratio, t_cycle

# Plant Config.
WV = 1.32  # m^3, Working Volume

# Time config.
t_ratio = [4.2 / 100, 8.3 / 100, 37.5 / 100, 31.2 / 100, 2.1 / 100, 8.3 / 100, 2.1 / 100, 6.3 / 100]

dt = 0.002 / 24  # derivate time(day)
t_delta = dt * 10  # running time, 기존 10

# phase time
t_cycle = 12 / 24  # hour -> day, 12hr
t_memory1, t_memory2, t_memory3, t_memory4, t_memory5, t_memory6,t_memory7, t_memory8 = batch_time(t_cycle, t_ratio, t_delta)
t_last = t_cycle

# Batch type
# batch_type = [0,1,2,3,4]
# | 0: Filling phase
# | 1: Reaction phase
# | 2: Settling phase
# | 3: Draw phase
# | 4: idle phase

"""

    Basic phase sequencing:



    Phase No./      Feeding     Aeration    Mixing      Discharge/  Type

    length(%)                                           Wastage

    1 (4.2)         Yes         No          Yes         No          FLL/Rxn (ANX)  0  

    2 (8.3)         No          No          Yes         No          Rxn (ANX)
1        
    3 (37.5)        No          Yes         Yes         No          Rxn (AER)
1        
    4 (31.2)        No          No          Yes         No          Rxn (ANX)
1        
    5 (2.1)         No          Yes         Yes         No          Rxn (AER)
1        
    6 (8.3)         No          No          No          No          STL
      2  
    7 (2.1)         No          No          No          Yes         DRW
      3  
    8 (6.3)         No          Yes         No          No          IDL

     4   
    (ref. Pons et al. Tbl. 1) 

"""

DO_control_par = [5.0, 0.00035, 0.02 / 24, 2, 0, 240, 12, 2, 5, 0.005, DO_set(15)]

# PID tuning
# | PID parameters: optimized parameters, (hand tuning)
Kc = 100  # 10
tauI = 20  # 0.5
tauD = 0  # 0.00005


class SbrCnt1(gym.Env):
    """custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # action space: Continuous Do setpoints | range-> -1~1 (applicable variation of DO, need to be checked with the other Envs.

        self.action_space = spaces.Box(np.array([-1]), np.array([1]),
                                       dtype=np.float32)  # -0.05~0.05 으로 DO setpoint change

        # state space: Continuous, components | normalized
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, -1]), high=np.ones([5]) * 1.0,
            # low=np.array([0, 0,0,0,0,0,0]), high=np.ones([7])*1.0, #
            dtype=np.float32)  # 7가지 ASM components를 state로 선정함.

        # SBR parameters
        self.Spar = [0.24, 0.67, 0.08, 0.08, 0.06]  # (ref. BSM1 report Tbl. 2) # BSM report에 따름
        #       Ya    Yh    fp    ixb   ixp
        self.Kpar = [4.0, 10.0, 0.2, 0.5, 0.3, 0.8, 0.8, 3.0, 0.1, 0.5, 1.0, 0.05, 0.4,
                     0.05]  # (ref. BSM1 report Tbl. 3) # BSM report에 따름
        #      muhath  Ks  Koh  Kno  bh   tag etah   kh   Kx muhata Knh  ba   Koa   Ka
        self.DO_control_par = [5.0, 0.00035, 0.02 / 24, 2, 0, 240, 12, 2, 5, 0.005,
                               DO_set(15)]  # DO controller의 control parameter
        self.biomass_setpoint = 2700  # settling 시 reactor에 남겨지는 biomass 총량 값. => 이 값에 맞도록 Qw를 빼냄.
        self.Qeff = 0.66
        self.u_t_t = np.array(np.zeros([1, 466]))  # 먼지잘 모르겠음.

        """
         List of variables :
                    0: time, 2=Ss, 6=Xbh, 7=Xba, 9=So, 10=Sno, 11=Snh
                    (ref. BSM1 report Tbl. 1)
        """
        self.state_idx = [0, 8, 10]  # 각 변수에 따른 위치값 #self.state_idx = [0, 2, 5, 6, 8, 9, 10] #각 변수에 따른 위치값 #

        self.x_1 = np.array([0.5, 8.,
                             30])  # Normalization을 위한 초기 값.self.x_1 =  np.array([0.5, 5, 3000., 200., 8., 30., 30]) # Normalization을 위한 초기 값.

        self.x_2 = np.zeros([1, len(self.state_idx)])
        self.x_2 = np.zeros([1, len(self.state_idx)])

        self.t_t = []  # DeepRL code와 결합시켜 plotting하기 위한 변수, time trajectory
        self.x_t = []  # DeepRL code와 결합시켜 plotting하기 위한 변수, state trajectory
        self.state_t = []

    def reset(self):

        # Reset the SBR env.
        # | 1. Generate the influent
        # | 2. Reset the initial state
        # | 3. Reset the time, batch_type
        # | 4. Calculate the DeepRL state
        # | 5. Volume update for running SBR

        global influent_mixed

        # 1. Generate the influent
        switch, influent_mixed, influent_var = buffer_tank.influent.buffer_tank(0)  # np.random.choice(8, 1))
        # Switch : number of generated influent scenarios
        # influent_mixed : Scalar values of state components
        # influent_var : time-varying values of stste components

        """
         List of variables for "influent_mixed" :
                    0=V, 1=Si, 2=Ss, 3=Xi, 4=Xs, 5=Xbh, 6=Xba, 7=Xp, 8=So, 9=Sno, 10=Snh, 11=Snd, 12=Xnd, 13=Salk
                    (ref. BSM1 report Tbl. 1)
        """

        global WV, IV
        global x0, x0_init
        global IV_init, Qin

        # 2. Reset the initial state: Stablized from SBR 100days
        # initial Inoculum Volume
        IV_init = 0.6161484733495801  # m^3, Inoculum Volume

        # Initial State

        x0_init = [0.6161484733495801, 30, 0.571098000538576, 1440.01157895393,
                   31.254221999137, 2599.2714348941, 168.915006750837, 551.901552960823, 2.16607843793004,
                   13.3791460027604, 0.00562880208518134, 0.35996687629947, 1.86916737961228, 3.790463057094611]

        # 3. Reset the time, batch_type

        global t, u, dcv, ie, e

        t = 0

        u = 0

        # | ii) parameters and  trajectory values related to PID control #len(t_save2) - 1
        dcv = []  # derivate of controlled variable
        ie = []  # integral of the error
        e = []  # error

        # | iii) trajectory values related to Oxygen
        global So, Kla
        So = []  # So trajectory
        Kla = []  # Kla trajectory

        # | iii) variables related to Oxygen
        kla = 0

        # | iV) trajectory related to time
        global t_t, x_t, u_t, state_t, So_t
        t_t = [0]  # time

        # | V) trajectory related to State
        # It has represented in the STEP as update
        x_t = []
        u_t = []
        So_t = []

        state_t = []

        # 4. Volume update
        # | Volume update from "Step"
        if 'IV_new' in globals():
            IV = IV_init  # IV_new
        if 'IV_new' not in globals():
            IV = IV_init
        # Initial states from "Step"
        if 'x0_new' in globals():
            x0 = x0_init  # x0_new
        if 'x0_new' not in globals():
            x0 = x0_init

        global Qin
        Qin = WV - IV

        # Assign: x_in
        if t == 0:
            x_in = x0  # when it is the first time to run SBR system (x from the reset stage)
        # | initialize DO conc.
        So.append(x_in[8])
        # | initialize Kla
        Kla.append(kla)
        x_t = np.array(x0)

        influent_mixed[0] = Qin / t_memory1[-1]  # [m^3 / filling time]

        global x_out

        # Filling phase
        x_out, t_range, Kla, So, dcv, ie, e = self.Sim_filling(x_in, t, t_ratio[0], influent_mixed, Kla, So, dcv, ie, e)
        x_t = np.vstack([x_t, x_out[1:]])  # stack the 'state trajectory', x_t
        t_range_f = t_range.tolist()
        t_t = t_t + t_range_f[1:]  # stack the 'time trajectory', t_t
        # print("t_t in reset: {}".format(np.shape(t_t)))
        # print("t_t in reset: {}".format((t_t)))
        # print("t_range_f in reset: {}".format(np.shape(t_range_f[1:])))

        x_out_1 = x_out[-1]  # the last state

        # So = x_t[::10, 8].tolist()
        Kla = Kla * (int(len(So) / len(Kla)))  # ??

        t = t_t[-1]  # the last time

        """
         List of variables :
                    0: time, 1=Ss, 4=Xbh, 5=Xba, 7=So, 8=Sno, 9=Snh, 10=Snd
                    (ref. BSM1 report Tbl. 1)
        """

        # 5. Calculate the DeepRL state
        for i in self.state_idx:
            if i == 0:
                self.x_2[0][0] = t_t[-1]  # time
            else:
                self.x_2[0][self.state_idx.index(i)] = (Qin * influent_mixed[i] + x_out_1[i] * IV) / (Qin + IV)
        state = self.x_2 / self.x_1

        state = state[0]  # 추후 삭제

        # 상태변화량, xdot
        xdot_so = (x_out[-1][8] - x_out[0][8]) / 8
        xdot_snh = (x_out[-1][10] - x_out[0][10]) / 20

        if xdot_so > 1:
            xdot_so = 1
        elif xdot_so < -1:
            xdot_so = -1

        if xdot_snh > 1:
            xdot_snh = 1
        elif xdot_snh < -1:
            xdot_snh = -1

        state = np.append(state, [xdot_so, xdot_snh])

        state_t.append(state)

        return state

    def step(self, action):
        global t, u, x_out, So, Kla, kla, dcv, ie, e, t_t, x_t, Qw, eff_component, u_t, state_t, So_t
        done = False  # Done will be True when reaction phases were finished(<=> Settling, drawing and idle phases would be started)

        if t < t_memory2[0]:  # Phs2(ANX)에 대한 시뮬레이션

            x_in = x_out[-1]

            x_out1, t_range1, Qw, PE, SP, EQI, eff_component = self.Sim_rxn(x_in, t_memory2, 0, [0], So, dcv, ie,
                                                                            e)  # Kla

            x_in = x_out1[-1]

            x_t = np.vstack([x_t, x_out1[1:]])
            t_t = t_t + t_range1[1:]

            x_in = x_t[-1]  # Update the initial value for running the SBR system

            t = t_t[-1]

        # Run the SBR system with "CONTINUOUS CONTROL"
        # input: t, u, x
        # | t: time variable at 'now' in running SBR phases
        # | u: generated setpoints by DeepRL agent
        # | x: State
        # | influent_mixed

        # setpoint: previous setpoint + Generated action of DeepRL agent
        u = u + action

        # print("action: {}".format(action))

        if u < 0:
            u = 0
        elif u > 8:
            u = 8
        else:
            u = u

        # u = 2
        # print("u : {}".format(u))

        x_in = x_t[-1]  # Update the initial value for running the SBR system

        t, x_out, t_t, x_t = self.run_step(t, u, x_in, influent_mixed, done)
        # print("t after run_step: {}".format(t))
        So_t = So_t + x_out[:, 8].tolist()

        for i in range(len(x_out[:, 8].tolist())):
            if type(u) == int:
                u_t.append(u)
            else:
                u_t.append(u[0])

        if (t >= t_memory3[-1]) & (t < t_memory4[-1]):  # Phs2(ANX)에 대한 시뮬레이션

            x_in = x_out[-1]

            x_out1, t_range1, Qw, PE, SP, EQI, eff_component = self.Sim_rxn(x_in, t_memory4, 0, [0], So, dcv, ie, e)

            x_in = x_out1[-1]

            x_t = np.vstack([x_t, x_out1[1:]])
            t_t = t_t + t_range1[1:]

            x_in = x_t[-1]  # Update the initial value for running the SBR system

            t = t_t[-1]

        reward = sbr_reward(x_out[-1], u_t, done, 0)

        # print("reward: {}".format(reward))
        # print("So: {}".format(So[-1]))

        # Next state
        x_in = x_out[-1]

        # Calculate state
        for i in self.state_idx:
            if i == 0:
                self.x_2[0][0] = t_t[-1]  # time
            else:
                self.x_2[0][self.state_idx.index(i)] = x_in[i]
        state = self.x_2 / self.x_1

        state = state[0]  # 추후 삭제

        # 상태변화량, xdot        
        xdot_so = (x_out[-1][8] - x_out[0][8]) / 8
        xdot_snh = (x_out[-1][10] - x_out[0][10]) / 20

        if xdot_so > 1:
            xdot_so = 1
        elif xdot_so < -1:
            xdot_so = -1

        if xdot_snh > 1:
            xdot_snh = 1
        elif xdot_snh < -1:
            xdot_snh = -1

        state = np.append(state, [xdot_so, xdot_snh])
        state_t.append(state)

        # print("u in step: {}".format(u))

        if t >= t_memory5[-1]:
            done = True

            x_out1, t_range1, Qw, PE, SP, EQI, eff_component = self.Sim_Settling_Drawing(x_in, t, t_ratio[5],
                                                                                         t_ratio[6], t_delta, self.Qeff,
                                                                                         self.biomass_setpoint)
            x_in = x_out1[-1]
            x_out2, t_range2, Kla, So, dcv, ie, e = self.Sim_idle(x_in, t_range1, t_ratio[7], u, Kla, So, dcv, ie,
                                                                  e)
            # print("u in IDL: {}".format(u))
            x_out = np.vstack([x_out1, x_out2[1:]])
            t_range1 = t_range1.tolist()
            t_range2 = t_range2.tolist()
            t_range = t_range1 + t_range2[1:]
            t_range = np.array(t_range)
            t_end = t_range[-1]
            t_instant = t_range.tolist()

            x_t = np.vstack([x_t, x_out[1:]])
            t_t = t_t + t_instant[1:]

            # reward = sbr_reward( x_out[-1], u_t,  done, eff_component)

            """x_in = x_out2[-1]

            for i in self.state_idx:
                if i == 0:
                    self.x_2[0][0] = t_t[-1]  # time
                else:
                    self.x_2[0][self.state_idx.index(i)] = x_in[i]
            state = self.x_2 / self.x_1

            state = state[0]  # 추후 삭제

            #self.u_t_t = np.vstack([self.u_t_t, u_t])

            #print("t after run_step in IDL: {}".format(t_t[-1]))"""

        return state, reward, done, {}

    def trajectory(self, ):
        global t, u, x_out, So, Kla, kla, dcv, ie, e, t_t, x_t, Qw, eff_component, u_t, state_t, So_t

        # print("in env, x_t: {}".format(np.shape(x_t)))
        # print("in env, t_t: {}".format(np.shape(t_t)))

        self.t_t = t_t  # DeepRL code와 결합시켜 plotting하기 위한 변수, time trajectory
        self.x_t = x_t  # DeepRL code와 결합시켜 plotting하기 위한 변수, state trajectory

        self.state_t = state_t

        # print("in env, x_t: {}".format(len(self.x_t)))

        return t_t, x_t, u_t, state_t, So_t

    def run_step(self, t, u, x_in, influent_mixed, done):  # steps for rxn phases
        global So, Kla, kla, dcv, ie, e, t_t, x_t, Qw, eff_component

        # Call: time range
        # | Controller time interval: t_start ~ t_end(t_start + t_delta)
        t_start = t  # start time to run the SBR system (ODE)
        t_end = t + t_delta  # end time when the SBR running has finished (ODE)
        # | derivative time interval: ~dt (t_delta/10)
        t_range = np.linspace(t_start, t_end, int((t_end - t_start) / dt))
        # Call: Simulate the SBR system/each phases have been divided
        # | 2. Reaction phase

        x_out, t_range, Kla, So, dcv, ie, e = self.Sim_rxn(x_in, t_range, u, Kla, So, dcv, ie, e)

        # | 2. time
        t = t_range[-1]
        # | update the temporal trajectories
        t_instant = t_range.tolist()

        if t == 0:
            t_t = t_t + t_instant
        else:
            t_t = t_t + t_instant[1:]  # change np array to list

        if t == 0:
            x_t = np.array(x_out)
        else:
            x_t = np.vstack([x_t, x_out[1:]])

        # | 3. initial value
        # It will be updated in the section of STEP

        return t, x_out, t_t, x_t

    # Call: ODE eqns. of each batch_type
    # | 1. Filling phase

    def filling_dxdt(self, x, t, Spar, Kpar, DO_control_par, Kla, loading):
        # Process 1
        rho1 = Kpar[0] * (x[2] / (Kpar[1] + x[2])) * (x[8] / (Kpar[2] + x[8])) * x[5]

        # Process 2
        rho2 = Kpar[0] * (x[2] / (Kpar[1] + x[2])) * (Kpar[2] / (x[8] + Kpar[2])) * (x[9] / (Kpar[3] + x[9])) * \
               Kpar[
                   5] * x[5]

        # Process 3
        rho3 = Kpar[9] * (x[10] / (Kpar[10] + x[10])) * (x[8] / (Kpar[12] + x[8])) * x[6]

        # Process 4
        rho4 = Kpar[4] * x[5]

        # Process 5
        rho5 = Kpar[11] * x[6]

        # Process 6
        rho6 = Kpar[13] * x[11] * x[5]

        # Process 7
        rho7 = Kpar[7] * ((x[4] / x[5]) / (Kpar[8] + (x[4] / x[5]))) * (
                (x[8] / (Kpar[2] + x[8])) + Kpar[6] * (Kpar[2] / (x[8] + Kpar[2])) * (
                x[9] / (Kpar[3] + x[9]))) * x[5]

        # Process 8
        rho8 = (x[12] / x[4]) * rho7

        # Stoichiometric Coeff.

        nu2_1 = -1 / Spar[1]
        nu5_1 = 1
        nu8_1 = -(1 - Spar[1]) / Spar[1]
        nu10_1 = -Spar[3]
        nu13_1 = -Spar[3] / 14

        nu2_2 = -1 / Spar[1]
        nu5_2 = 1
        nu9_2 = -((1 - Spar[1]) / (2.86 * Spar[1]))
        nu10_2 = -Spar[3]
        nu13_2 = (1 - Spar[1]) / (14 * 2.86 * Spar[1]) - Spar[3] / 14

        nu6_3 = 1
        nu8_3 = -(4.57 - Spar[0]) / Spar[0]
        nu9_3 = 1 / Spar[0]
        nu10_3 = -Spar[3] - 1 / Spar[0]
        nu13_3 = -Spar[3] / 14 - 1 / (7 * Spar[0])

        nu4_4 = 1 - Spar[4]
        nu5_4 = -1
        nu7_4 = Spar[4]
        nu12_4 = Spar[3] - Spar[2] * Spar[4]

        nu4_5 = 1 - Spar[4]
        nu6_5 = -1
        nu7_5 = Spar[4]
        nu12_5 = Spar[3] - Spar[2] * Spar[4]

        nu10_6 = 1
        nu11_6 = -1
        nu13_6 = 1 / 14

        nu2_7 = 1
        nu4_7 = -1

        nu11_8 = 1
        nu12_8 = -1

        kla = Kla

        # 0 = Si,
        r1 = 0
        # 1 = Ss,
        r2 = nu2_1 * rho1 + nu2_2 * rho2 + nu2_7 * rho7
        # 2 = Xi,
        r3 = 0
        # 3 = Xs,
        r4 = nu4_4 * rho4 + nu4_5 * rho5 + nu4_7 * rho7
        # 4 = Xbh,
        r5 = nu5_1 * rho1 + nu5_2 * rho2 + nu5_4 * rho4
        # 5 = Xba,
        r6 = nu6_3 * rho3 + nu6_5 * rho5
        # 6 = Xp,
        r7 = nu7_4 * rho4 + nu7_5 * rho5
        # 7 = So,
        r8 = nu8_1 * rho1 + nu8_3 * rho3 + kla * (DO_control_par[10] - x[8])
        # 8 = Sno,
        r9 = nu9_2 * rho2 + nu9_3 * rho3
        # 9 = Snh,
        r10 = nu10_1 * rho1 + nu10_2 * rho2 + nu10_3 * rho3 + nu10_6 * rho6
        # 10 = Snd,
        r11 = nu11_6 * rho6 + nu11_8 * rho8
        # 11 = Xnd,
        r12 = nu12_4 * rho4 + nu12_5 * rho5 + nu12_8 * rho8
        # 12 = Salk
        r13 = nu13_1 * rho1 + nu13_2 * rho2 + nu13_3 * rho3 + nu13_6 * rho6

        dxdt = np.zeros_like(x)
        # 0 = Working volume,
        dxdt[0] = loading[0]
        # 1 = Si,
        dxdt[1] = r1 + (loading[0] / x[0]) * (loading[1] - x[1])
        # 2 = Ss,
        dxdt[2] = r2 + (loading[0] / x[0]) * (loading[2] - x[2])
        # 3 = Xi,
        dxdt[3] = r3 + (loading[0] / x[0]) * (loading[3] - x[3])
        # 4 = Xs,
        dxdt[4] = r4 + (loading[0] / x[0]) * (loading[4] - x[4])
        # 5 = Xbh,
        dxdt[5] = r5 + (loading[0] / x[0]) * (loading[5] - x[5])
        # 6 = Xba,
        dxdt[6] = r6 + (loading[0] / x[0]) * (loading[6] - x[6])
        # 7 = Xp,
        dxdt[7] = r7 + (loading[0] / x[0]) * (loading[7] - x[7])
        # 8 = So,
        dxdt[8] = r8 + (loading[0] / x[0]) * (loading[8] - x[8])
        # 9 = Sno,
        dxdt[9] = r9 + (loading[0] / x[0]) * (loading[9] - x[9])
        # 10 = Snh,
        dxdt[10] = r10 + (loading[0] / x[0]) * (loading[10] - x[10])
        # 11 = Snd,
        dxdt[11] = r11 + (loading[0] / x[0]) * (loading[11] - x[11])
        # 12 = Xnd,
        dxdt[12] = r12 + (loading[0] / x[0]) * (loading[12] - x[12])
        # 13 = Salk
        dxdt[13] = r13 + (loading[0] / x[0]) * (loading[13] - x[13])

        return dxdt

    def Sim_filling(self, x, t, t_filling, loading, Kla, So, dcv, ie, e):
        t_start = t
        t_end = t + t_filling * 0.5
        t_range = np.linspace(t_start, t_end, int((t_end - t_start) / dt))

        # Call: Generate Setpoints via PID controllers from u
        # | Set-point
        sp = 0  # u # set-points

        e.append(sp - So[-1])

        if t_start > 0:  # calcuate starting on second cycle
            dcv.append((So[-1] - So[-2]) / dt)
            ie.append(ie[-1] + e[-1] * dt)
        else:
            dcv.append(0)
            ie.append(0)

        P = Kc * e[-1]
        I = Kc / tauI * ie[-1]
        D = Kc * tauD * dcv[-1]

        Kla.append(P + I + D + Kla[-1])  # Kla : Bias <- how estimate?

        if Kla[-1] > DO_control_par[5]:  # check upper limit
            Kla[-1] = DO_control_par[5]
            ie[-1] = ie[-1] - e[-1] * dt  # anti-reset windup
        if Kla[-1] < DO_control_par[4]:  # check lower limit
            Kla[-1] = DO_control_par[4]
            ie[-1] = ie[-1] - e[-1] * dt  # anti-reset windup

        x_out = integrate.odeint(self.filling_dxdt, x, t_range,
                                 args=(self.Spar, self.Kpar, self.DO_control_par, Kla[-1], loading,))

        So.append(x_out[-1][8])

        return x_out, t_range, Kla, So, dcv, ie, e

    # | 2. Reaction phase

    def reaction_dxdt(self, x, t, Spar, Kpar, DO_control_par, Kla):
        # Process 1
        rho1 = Kpar[0] * (x[2] / (Kpar[1] + x[2])) * (x[8] / (Kpar[2] + x[8])) * x[5]

        # Process 2
        rho2 = Kpar[0] * (x[2] / (Kpar[1] + x[2])) * (Kpar[2] / (x[8] + Kpar[2])) * (x[9] / (Kpar[3] + x[9])) * \
               Kpar[
                   5] * x[5]

        # Process 3
        rho3 = Kpar[9] * (x[10] / (Kpar[10] + x[10])) * (x[8] / (Kpar[12] + x[8])) * x[6]

        # Process 4
        rho4 = Kpar[4] * x[5]

        # Process 5
        rho5 = Kpar[11] * x[6]

        # Process 6
        rho6 = Kpar[13] * x[11] * x[5]

        # Process 7
        rho7 = Kpar[7] * ((x[4] / x[5]) / (Kpar[8] + (x[4] / x[5]))) * (
                (x[8] / (Kpar[2] + x[8])) + Kpar[6] * (Kpar[2] / (x[8] + Kpar[2])) * (x[9] / (Kpar[3] + x[9]))) * x[
                   5]

        # Process 8
        rho8 = (x[12] / x[4]) * rho7

        # Stoichiometric Coeff.

        nu2_1 = -1 / Spar[1]
        nu5_1 = 1
        nu8_1 = -(1 - Spar[1]) / Spar[1]
        nu10_1 = -Spar[3]
        nu13_1 = -Spar[3] / 14

        nu2_2 = -1 / Spar[1]
        nu5_2 = 1
        nu9_2 = -((1 - Spar[1]) / (2.86 * Spar[1]))
        nu10_2 = -Spar[3]
        nu13_2 = (1 - Spar[1]) / (14 * 2.86 * Spar[1]) - Spar[3] / 14

        nu6_3 = 1
        nu8_3 = -(4.57 - Spar[0]) / Spar[0]
        nu9_3 = 1 / Spar[0]
        nu10_3 = -Spar[3] - 1 / Spar[0]
        nu13_3 = -Spar[3] / 14 - 1 / (7 * Spar[0])

        nu4_4 = 1 - Spar[4]
        nu5_4 = -1
        nu7_4 = Spar[4]
        nu12_4 = Spar[3] - Spar[2] * Spar[4]

        nu4_5 = 1 - Spar[4]
        nu6_5 = -1
        nu7_5 = Spar[4]
        nu12_5 = Spar[3] - Spar[2] * Spar[4]

        nu10_6 = 1
        nu11_6 = -1
        nu13_6 = 1 / 14

        nu2_7 = 1
        nu4_7 = -1

        nu11_8 = 1
        nu12_8 = -1

        kla = Kla

        # 0 = Si,
        r1 = 0
        # 1 = Ss,
        r2 = nu2_1 * rho1 + nu2_2 * rho2 + nu2_7 * rho7
        # 2 = Xi,
        r3 = 0
        # 3 = Xs,
        r4 = nu4_4 * rho4 + nu4_5 * rho5 + nu4_7 * rho7
        # 4 = Xbh,
        r5 = nu5_1 * rho1 + nu5_2 * rho2 + nu5_4 * rho4
        # 5 = Xba,
        r6 = nu6_3 * rho3 + nu6_5 * rho5
        # 6 = Xp,
        r7 = nu7_4 * rho4 + nu7_5 * rho5
        # 7 = So,
        r8 = nu8_1 * rho1 + nu8_3 * rho3 + kla * (DO_control_par[10] - x[8])
        # 8 = Sno,
        r9 = nu9_2 * rho2 + nu9_3 * rho3
        # 9 = Snh,
        r10 = nu10_1 * rho1 + nu10_2 * rho2 + nu10_3 * rho3 + nu10_6 * rho6
        # 10 = Snd,
        r11 = nu11_6 * rho6 + nu11_8 * rho8
        # 11 = Xnd,
        r12 = nu12_4 * rho4 + nu12_5 * rho5 + nu12_8 * rho8
        # 12 = Salk
        r13 = nu13_1 * rho1 + nu13_2 * rho2 + nu13_3 * rho3 + nu13_6 * rho6

        dxdt = np.zeros_like(x)
        # 0 = Working volume,
        dxdt[0] = 0
        # 1 = Si,
        dxdt[1] = r1
        # 2 = Ss,
        dxdt[2] = r2
        # 3 = Xi,
        dxdt[3] = r3
        # 4 = Xs,
        dxdt[4] = r4
        # 5 = Xbh,
        dxdt[5] = r5
        # 6 = Xba,
        dxdt[6] = r6
        # 7 = Xp,
        dxdt[7] = r7
        # 8 = So,
        dxdt[8] = r8
        # 9 = Sno,
        dxdt[9] = r9
        # 10 = Snh,
        dxdt[10] = r10
        # 11 = Snd,
        dxdt[11] = r11
        # 12 = Xnd,
        dxdt[12] = r12
        # 13 = Salk
        dxdt[13] = r13

        return dxdt

    def Sim_rxn(self, x, t_range, u, Kla, So, dcv, ie, e):

        # Call: time
        t_start = t_range[0]
        t_end = t_range[-1]

        # Call: Generate Setpoints via PID controllers from u
        # | Set-point
        sp = u  # set-points

        e.append(sp - So[-1])

        if t_start > 0:  # calcuate starting on second cycle
            dcv.append((So[-1] - So[-2]) / dt)
            ie.append(ie[-1] + e[-1] * dt)
        else:
            dcv.append(0)
            ie.append(0)

        P = Kc * e[-1]
        I = Kc / tauI * ie[-1]
        D = Kc * tauD * dcv[-1]

        Kla.append(P + I + D + Kla[-1])  # Kla : Bias <- how estimate?

        if Kla[-1] > DO_control_par[5]:  # check upper limit
            Kla[-1] = DO_control_par[5]
            ie[-1] = ie[-1] - e[-1] * dt  # anti-reset windup
        if Kla[-1] < DO_control_par[4]:  # check lower limit
            Kla[-1] = DO_control_par[4]
            ie[-1] = ie[-1] - e[-1] * dt  # anti-reset windup

        x_out = integrate.odeint(self.reaction_dxdt, x, t_range,
                                 args=(self.Spar, self.Kpar, self.DO_control_par, Kla[-1],))

        So.append(x_out[-1][8])

        return x_out, t_range, Kla, So, dcv, ie, e

    # | 3. Settling, Drawing phases

    def settling_dXnddt(self, Xnd, t, z, Xf):

        sX1 = Xnd[0]
        sX2 = Xnd[1]
        sX3 = Xnd[2]
        sX4 = Xnd[3]
        sX5 = Xnd[4]
        sX6 = Xnd[5]
        sX7 = Xnd[6]
        sX8 = Xnd[7]
        sX9 = Xnd[8]
        sX10 = Xnd[9]

        """
        % Settler data

        As : Area(m^2) Vs : Volume(m^3)
        """
        As = (1.25 / 2) ** 2  # 12.5 / 4

        # Assume: completely mixed

        """
        Flowrate
        In settling phase, it assumes there is no bulk movement(no Qin, Qeff, Qw)
        """
        Qin = 0
        Qr2 = 0
        Qw = 0
        Qeff = Qin - Qw
        Qf = Qr2 + Qeff + Qw

        """ 
       Settler Parameters

        vbnd: Maximum settling velocity, vmax: Maximum Vesilind settling velocity
        rh: Hindered zone settling parameter, rp: Flocculant zone settling parameter 
        fns: Non-settleable fraction
        """
        vbnd = 250
        vmax = 474
        rh = 0.000576
        rp = 0.00286
        fns = 0.00228

        vdn = (Qr2 + Qw) / As
        vup = Qeff / As
        Xt = 3000  # sludge blanket threshold concentration

        """
        Mass balance for settler

            Solid flux due to gravity
        """

        v_sX1 = max(0, min(vbnd, vmax * (np.exp(-rh * (sX1 - fns * Xf)) - np.exp(-rp * (sX1 - fns * Xf)))))
        v_sX2 = max(0, min(vbnd, vmax * (np.exp(-rh * (sX2 - fns * Xf)) - np.exp(-rp * (sX2 - fns * Xf)))))
        v_sX3 = max(0, min(vbnd, vmax * (np.exp(-rh * (sX3 - fns * Xf)) - np.exp(-rp * (sX3 - fns * Xf)))))
        v_sX4 = max(0, min(vbnd, vmax * (np.exp(-rh * (sX4 - fns * Xf)) - np.exp(-rp * (sX4 - fns * Xf)))))
        v_sX5 = max(0, min(vbnd, vmax * (np.exp(-rh * (sX5 - fns * Xf)) - np.exp(-rp * (sX5 - fns * Xf)))))
        v_sX6 = max(0, min(vbnd, vmax * (np.exp(-rh * (sX6 - fns * Xf)) - np.exp(-rp * (sX6 - fns * Xf)))))
        v_sX7 = max(0, min(vbnd, vmax * (np.exp(-rh * (sX7 - fns * Xf)) - np.exp(-rp * (sX7 - fns * Xf)))))
        v_sX8 = max(0, min(vbnd, vmax * (np.exp(-rh * (sX8 - fns * Xf)) - np.exp(-rp * (sX8 - fns * Xf)))))
        v_sX9 = max(0, min(vbnd, vmax * (np.exp(-rh * (sX9 - fns * Xf)) - np.exp(-rp * (sX9 - fns * Xf)))))
        v_sX10 = max(0, min(vbnd, vmax * (np.exp(-rh * (sX10 - fns * Xf)) - np.exp(-rp * (sX10 - fns * Xf)))))

        J_sX1 = v_sX1 * sX1
        J_sX2 = v_sX2 * sX2
        J_sX3 = v_sX3 * sX3
        J_sX4 = v_sX4 * sX4
        J_sX5 = v_sX5 * sX5
        J_sX6 = v_sX6 * sX6
        J_sX7 = v_sX7 * sX7

        if sX5 <= Xt:
            J_clar_X6 = v_sX6 * sX6
        else:
            J_clar_X6 = min(v_sX6 * sX6, v_sX5 * sX5)

        if sX6 <= Xt:
            J_clar_X7 = v_sX7 * sX7
        else:
            J_clar_X7 = min(v_sX7 * sX7, v_sX6 * sX6)

        if sX7 <= Xt:
            J_clar_X8 = v_sX8 * sX8
        else:
            J_clar_X8 = min(v_sX8 * sX8, v_sX7 * sX7)

        if sX8 <= Xt:
            J_clar_X9 = v_sX9 * sX9
        else:
            J_clar_X9 = min(v_sX9 * sX9, v_sX8 * sX8)

        if sX9 <= Xt:
            J_clar_X10 = v_sX10 * sX10
        else:
            J_clar_X10 = min(v_sX10 * sX10, v_sX9 * sX9)

        dsXdt = np.zeros_like(Xnd)

        # Particle component
        dsXdt[0] = (vdn * (Xnd[1] - Xnd[0]) + min(J_sX2, J_sX1)) / z
        dsXdt[1] = (vdn * (Xnd[2] - Xnd[1]) + min(J_sX2, J_sX3) - min(J_sX2, J_sX1)) / z
        dsXdt[2] = (vdn * (Xnd[3] - Xnd[2]) + min(J_sX3, J_sX4) - min(J_sX3, J_sX2)) / z
        dsXdt[3] = (vdn * (Xnd[4] - Xnd[3]) + min(J_sX4, J_sX5) - min(J_sX4, J_sX3)) / z
        dsXdt[4] = (vdn * (Xnd[5] - Xnd[4]) + min(J_sX5, J_sX6) - min(J_sX5, J_sX4)) / z
        dsXdt[5] = (vdn * (Xnd[6] - Xnd[5]) + min(J_sX6, J_sX7) - min(J_sX6, J_sX5)) / z + (
                vup * (Xnd[4] - Xnd[5]) + J_clar_X7 - J_clar_X6) / z
        dsXdt[6] = (vup * (Xnd[5] - Xnd[6]) + J_clar_X8 - J_clar_X7) / z
        dsXdt[7] = (vup * (Xnd[6] - Xnd[7]) + J_clar_X9 - J_clar_X8) / z
        dsXdt[8] = (vup * (Xnd[7] - Xnd[8]) + J_clar_X10 - J_clar_X9) / z
        dsXdt[9] = (vup * (Xnd[8] - Xnd[9]) - J_clar_X10) / z

        return dsXdt

    def settling_dsXdt(self, sX, t, z, Xf):

        sX1 = sX[0]
        sX2 = sX[1]
        sX3 = sX[2]
        sX4 = sX[3]
        sX5 = sX[4]
        sX6 = sX[5]
        sX7 = sX[6]
        sX8 = sX[7]
        sX9 = sX[8]
        sX10 = sX[9]

        """
        % Settler data

        As : Area(m^2) Vs : Volume(m^3)
        """
        As = (1.25 / 2) ** 2  # 12.5 / 4

        # Assume: completely mixed

        """
        Flowrate
        In settling phase, it assumes there is no bulk movement(no Qin, Qeff, Qw)
        """
        Qin = 0
        Qr2 = 0
        Qw = 0
        Qeff = Qin - Qw
        Qf = Qr2 + Qeff + Qw

        """ 
       Settler Parameters

        vbnd: Maximum settling velocity, vmax: Maximum Vesilind settling velocity
        rh: Hindered zone settling parameter, rp: Flocculant zone settling parameter 
        fns: Non-settleable fraction
        """
        vbnd = 250
        vmax = 474
        rh = 0.000576
        rp = 0.00286
        fns = 0.00228

        vdn = (Qr2 + Qw) / As
        vup = Qeff / As
        Xt = 3000  # sludge blanket threshold concentration

        """
        Mass balance for settler

            Solid flux due to gravity
        """

        v_sX1 = max(vmax, (np.exp(-rh * (sX1 - fns * Xf)) - np.exp(-rp * (sX1 - fns * Xf))))
        v_sX2 = max(vmax, (np.exp(-rh * (sX2 - fns * Xf)) - np.exp(-rp * (sX2 - fns * Xf))))
        v_sX3 = max(vmax, (np.exp(-rh * (sX3 - fns * Xf)) - np.exp(-rp * (sX3 - fns * Xf))))
        v_sX4 = max(vmax, (np.exp(-rh * (sX4 - fns * Xf)) - np.exp(-rp * (sX4 - fns * Xf))))
        v_sX5 = max(vmax, (np.exp(-rh * (sX5 - fns * Xf)) - np.exp(-rp * (sX5 - fns * Xf))))
        v_sX6 = max(vmax, (np.exp(-rh * (sX6 - fns * Xf)) - np.exp(-rp * (sX6 - fns * Xf))))
        v_sX7 = max(vmax, (np.exp(-rh * (sX7 - fns * Xf)) - np.exp(-rp * (sX7 - fns * Xf))))
        v_sX8 = max(vmax, (np.exp(-rh * (sX8 - fns * Xf)) - np.exp(-rp * (sX8 - fns * Xf))))
        v_sX9 = max(vmax, (np.exp(-rh * (sX9 - fns * Xf)) - np.exp(-rp * (sX9 - fns * Xf))))
        v_sX10 = max(vmax, (np.exp(-rh * (sX10 - fns * Xf)) - np.exp(-rp * (sX10 - fns * Xf))))

        J_sX1 = v_sX1 * sX1
        J_sX2 = v_sX2 * sX2
        J_sX3 = v_sX3 * sX3
        J_sX4 = v_sX4 * sX4
        J_sX5 = v_sX5 * sX5
        J_sX6 = v_sX6 * sX6
        J_sX7 = v_sX7 * sX7
        J_sX8 = v_sX8 * sX8
        J_sX9 = v_sX9 * sX9
        J_sX10 = v_sX10 * sX10

        dsXdt = np.zeros_like(sX)

        # Particle component
        dsXdt[0] = J_sX2 / z
        dsXdt[1] = (J_sX3 - J_sX2) / z
        dsXdt[2] = (J_sX4 - J_sX3) / z
        dsXdt[3] = (J_sX5 - J_sX4) / z
        dsXdt[4] = (J_sX6 - J_sX5) / z
        dsXdt[5] = (J_sX7 - J_sX6) / z
        dsXdt[6] = (J_sX8 - J_sX7) / z
        dsXdt[7] = (J_sX9 - J_sX8) / z
        dsXdt[8] = (J_sX10 - J_sX9) / z
        dsXdt[9] = (0 - J_sX10) / z

        return dsXdt

    def Sim_Settling_Drawing(self, x, t, t_settling, t_drawing, dt, Qeff, biomass_setpoint):

        global Qw, So

        # Settling phases
        t_range_settling = np.linspace(t, t + t_settling * t_cycle, int((t_settling * t_cycle) / dt))

        x0 = x[0]  # V

        x5 = x[5]  # Xh
        x6 = x[6]  # Xa
        x3 = x[3]  # Xi
        x4 = x[4]  # Xs
        x7 = x[7]  # Xp
        x12 = x[12]  # Xnd

        Xf = 0.75 * (x3 + x4 + x5 + x6 + x7)  # + Xp5 + Xi5 # sludge conc.  from final reactor

        """
        % Settler data

        As : Area(m^2) Vs : Volume(m^3)
        """
        As = (1.25 / 2) ** 2  # 12.5 / 4
        Vs = x0
        z = Vs / As

        # Assume: completely mixed

        Xnd1 = x12 / 10
        Xnd2 = x12 / 10
        Xnd3 = x12 / 10
        Xnd4 = x12 / 10
        Xnd5 = x12 / 10
        Xnd6 = x12 / 10
        Xnd7 = x12 / 10
        Xnd8 = x12 / 10
        Xnd9 = x12 / 10
        Xnd10 = x12 / 10

        sX1 = Xf
        sX2 = Xf
        sX3 = Xf
        sX4 = Xf
        sX5 = Xf
        sX6 = Xf
        sX7 = Xf
        sX8 = Xf
        sX9 = Xf
        sX10 = Xf

        Xnd = [Xnd1, Xnd2, Xnd3, Xnd4, Xnd5, Xnd6, Xnd7, Xnd8, Xnd9, Xnd10]
        sX = [sX1, sX2, sX3, sX4, sX5, sX6, sX7, sX8, sX9, sX10]

        dXnddt = integrate.odeint(self.settling_dXnddt, Xnd, t_range_settling, args=(z, Xf,))
        dsXdt = integrate.odeint(self.settling_dsXdt, sX, t_range_settling, args=(z, Xf,))

        Xnd = dXnddt[-1]
        sX = dsXdt[-1]

        x_settling = np.array(x.tolist() * len(t_range_settling))
        x_settling.resize(len(t_range_settling), len(x))

        # Drawing phases

        t_range_drawing = np.linspace(t_range_settling[-1], t_range_settling[-1] + t_drawing * t_cycle,
                                      int((t_drawing * t_cycle) / dt))

        layer_volume = Vs / 10
        residual_V = Vs - Qeff

        # 1. effluent drawing(유출수 방출)

        # 방류수로 방출될 layer 갯수 선정
        m = int(math.ceil(round(Qeff / (layer_volume))))

        # effluent로 방출되는 슬러지
        sX_eff = sum(sX[-m:-1] * layer_volume)

        X_eff = copy.deepcopy(x[:])

        X_eff[0] = Qeff
        X_eff[4] = X_eff[4] * (1 / 0.75) * sX_eff / Xf
        X_eff[7] = X_eff[7] * (1 / 0.75) * sX_eff / Xf
        X_eff[3] = X_eff[3] * (1 / 0.75) * sX_eff / Xf
        X_eff[5] = X_eff[5] * (1 / 0.75) * sX_eff / Xf
        X_eff[6] = X_eff[6] * (1 / 0.75) * sX_eff / Xf

        # 방류수로 방출되고 남은 sX와 레이어 (10-m)
        residual_sX = sX[0:10 - m]
        residual_sX_weight_inLayer = layer_volume * residual_sX
        residual_sX_weight = sum(residual_sX_weight_inLayer)

        desire_sX_weight = biomass_setpoint * (residual_V)
        waste_sX_weight = residual_sX_weight - desire_sX_weight

        for i in range(10 - m):

            m_q = i

            residual_sX_weight_by_discharging = waste_sX_weight - residual_sX_weight_inLayer[i]

            if residual_sX_weight_by_discharging > 0:
                waste_sX_weight = residual_sX_weight_by_discharging
                residual_sX[i] = 0
                residual_sX_weight_inLayer[i] = 0
                residual_V -= layer_volume

            else:
                Qw = waste_sX_weight / (residual_sX[i] - biomass_setpoint)

                residual_sX_weight_inLayer[i] = residual_sX_weight_inLayer[i] - Qw * residual_sX[i]
                residual_V -= Qw
                residual_sX[i] = residual_sX_weight_inLayer[i] / (layer_volume - Qw)

                break

        sX2 = (sum(residual_sX_weight_inLayer)) / residual_V

        x_n = copy.deepcopy(x[:])

        x_n[0] = residual_V
        x_n[4] = x[4] * (1 / 0.75) * sX2 / Xf
        x_n[7] = x[7] * (1 / 0.75) * sX2 / Xf
        x_n[3] = x[3] * (1 / 0.75) * sX2 / Xf
        x_n[5] = x[5] * (1 / 0.75) * sX2 / Xf
        x_n[6] = x[6] * (1 / 0.75) * sX2 / Xf

        # soluable component

        # particulate component

        # PE: Pumping energy, (kWh/d)


        PE = 0.05 * abs(Qw)

        # sludge production, (Kg/d)
        SP = sX_eff * Qeff

        EQI, eff_component = self.cal_eq(sX_eff, X_eff, waste_sX_weight, Qw)

        t_range_settling_list = t_range_settling.tolist()
        t_range_drawing_list = t_range_drawing.tolist()
        t_range = np.array(t_range_settling_list + t_range_drawing_list[1:])

        x_drawing = np.array(x_n.tolist() * len(t_range_drawing_list))
        x_drawing.resize(len(t_range_drawing_list), len(x_n))
        x_out = np.vstack([x_settling, x_drawing[1:]])


        So_out = x_out[:, 8].tolist()
        So += So_out

        return x_out, t_range, Qw, PE, SP, EQI, eff_component

    # | 5. Idle phase

    def idle_dxdt(self, x, t, Spar, Kpar, DO_control_par, Kla):
        # Process 1
        rho1 = Kpar[0] * (x[2] / (Kpar[1] + x[2])) * (x[8] / (Kpar[2] + x[8])) * x[5]

        # Process 2
        rho2 = Kpar[0] * (x[2] / (Kpar[1] + x[2])) * (Kpar[2] / (x[8] + Kpar[2])) * (x[9] / (Kpar[3] + x[9])) * \
               Kpar[
                   5] * x[5]

        # Process 3
        rho3 = Kpar[9] * (x[10] / (Kpar[10] + x[10])) * (x[8] / (Kpar[12] + x[8])) * x[6]

        # Process 4
        rho4 = Kpar[4] * x[5]

        # Process 5
        rho5 = Kpar[11] * x[6]

        # Process 6
        rho6 = Kpar[13] * x[11] * x[5]

        # Process 7
        rho7 = Kpar[7] * ((x[4] / x[5]) / (Kpar[8] + (x[4] / x[5]))) * (
                (x[8] / (Kpar[2] + x[8])) + Kpar[6] * (Kpar[2] / (x[8] + Kpar[2])) * (x[9] / (Kpar[3] + x[9]))) * x[
                   5]

        # Process 8
        rho8 = (x[12] / x[4]) * rho7

        # Stoichiometric Coeff.

        nu2_1 = -1 / Spar[1]
        nu5_1 = 1
        nu8_1 = -(1 - Spar[1]) / Spar[1]
        nu10_1 = -Spar[3]
        nu13_1 = -Spar[3] / 14

        nu2_2 = -1 / Spar[1]
        nu5_2 = 1
        nu9_2 = -((1 - Spar[1]) / (2.86 * Spar[1]))
        nu10_2 = -Spar[3]
        nu13_2 = (1 - Spar[1]) / (14 * 2.86 * Spar[1]) - Spar[3] / 14

        nu6_3 = 1
        nu8_3 = -(4.57 - Spar[0]) / Spar[0]
        nu9_3 = 1 / Spar[0]
        nu10_3 = -Spar[3] - 1 / Spar[0]
        nu13_3 = -Spar[3] / 14 - 1 / (7 * Spar[0])

        nu4_4 = 1 - Spar[4]
        nu5_4 = -1
        nu7_4 = Spar[4]
        nu12_4 = Spar[3] - Spar[2] * Spar[4]

        nu4_5 = 1 - Spar[4]
        nu6_5 = -1
        nu7_5 = Spar[4]
        nu12_5 = Spar[3] - Spar[2] * Spar[4]

        nu10_6 = 1
        nu11_6 = -1
        nu13_6 = 1 / 14

        nu2_7 = 1
        nu4_7 = -1

        nu11_8 = 1
        nu12_8 = -1

        kla = Kla

        # 0 = Si,
        r1 = 0
        # 1 = Ss,
        r2 = nu2_1 * rho1 + nu2_2 * rho2 + nu2_7 * rho7
        # 2 = Xi,
        r3 = 0
        # 3 = Xs,
        r4 = nu4_4 * rho4 + nu4_5 * rho5 + nu4_7 * rho7
        # 4 = Xbh,
        r5 = nu5_1 * rho1 + nu5_2 * rho2 + nu5_4 * rho4
        # 5 = Xba,
        r6 = nu6_3 * rho3 + nu6_5 * rho5
        # 6 = Xp,
        r7 = nu7_4 * rho4 + nu7_5 * rho5
        # 7 = So,
        r8 = nu8_1 * rho1 + nu8_3 * rho3 + kla * (DO_control_par[10] - x[8])
        # 8 = Sno,
        r9 = nu9_2 * rho2 + nu9_3 * rho3
        # 9 = Snh,
        r10 = nu10_1 * rho1 + nu10_2 * rho2 + nu10_3 * rho3 + nu10_6 * rho6
        # 10 = Snd,
        r11 = nu11_6 * rho6 + nu11_8 * rho8
        # 11 = Xnd,
        r12 = nu12_4 * rho4 + nu12_5 * rho5 + nu12_8 * rho8
        # 12 = Salk
        r13 = nu13_1 * rho1 + nu13_2 * rho2 + nu13_3 * rho3 + nu13_6 * rho6

        dxdt = np.zeros_like(x)
        # 0 = Working volume,
        dxdt[0] = 0
        # 1 = Si,
        dxdt[1] = r1
        # 2 = Ss,
        dxdt[2] = r2
        # 3 = Xi,
        dxdt[3] = r3
        # 4 = Xs,
        dxdt[4] = r4
        # 5 = Xbh,
        dxdt[5] = r5
        # 6 = Xba,
        dxdt[6] = r6
        # 7 = Xp,
        dxdt[7] = r7
        # 8 = So,
        dxdt[8] = r8
        # 9 = Sno,
        dxdt[9] = r9
        # 10 = Snh,
        dxdt[10] = r10
        # 11 = Snd,
        dxdt[11] = r11
        # 12 = Xnd,
        dxdt[12] = r12
        # 13 = Salk
        dxdt[13] = r13

        return dxdt

    def Sim_idle(self, x, t_range, t_idle, u, Kla, So, dcv, ie, e):

        # Call: time
        t_start = t_range[-1]
        t_end = t_cycle  # t_start + t_idle*t_cycle
        t_range_idle = np.linspace(t_start, t_end, int((t_end - t_start) / dt))

        # Call: Generate Setpoints via PID controllers from u
        # | Set-point
        sp = u  # set-points

        e.append(sp - So[-1])

        if t_start > 0:  # calcuate starting on second cycle
            dcv.append((So[-1] - So[-2]) / dt)
            ie.append(ie[-1] + e[-1] * dt)
        else:
            dcv.append(0)
            ie.append(0)

        P = Kc * e[-1]
        I = Kc / tauI * ie[-1]
        D = Kc * tauD * dcv[-1]

        Kla.append(P + I + D + Kla[-1])  # Kla : Bias <- how estimate?

        if Kla[-1] > DO_control_par[5]:  # check upper limit
            Kla[-1] = DO_control_par[5]
            ie[-1] = ie[-1] - e[-1] * dt  # anti-reset windup
        if Kla[-1] < DO_control_par[4]:  # check lower limit
            Kla[-1] = DO_control_par[4]
            ie[-1] = ie[-1] - e[-1] * dt  # anti-reset windup

        x_out = integrate.odeint(self.idle_dxdt, x, t_range_idle,
                                 args=(self.Spar, self.Kpar, self.DO_control_par, Kla[-1],))

        So.append(x_out[-1][8])

        return x_out, t_range_idle, Kla, So, dcv, ie, e

    def cal_eq(self, sX_eff, x_eff, waste_sX_weight, Qw):

        V = x_eff[0]
        Si = x_eff[1]
        Ss = x_eff[2]
        Xi = x_eff[3]
        Xs = x_eff[4]
        Xbh = x_eff[5]
        Xba = x_eff[6]
        Xp = x_eff[7]
        So = x_eff[8]
        Sno = x_eff[9]
        Snh = x_eff[10]
        Snd = x_eff[11]
        Xnd = x_eff[12]
        Salk = x_eff[13]

        # Kinetic parameter
        i_xb = 0.08
        i_xp = 0.06
        fp = 0.08

        # EQI
        # weighting factor
        B_ss = 2
        B_COD = 1
        B_NKj = 30
        B_NO = 10
        B_BOD = 2

        Snkj = Snh + Snd + Xnd + i_xb * (Xbh + Xba) + i_xp * (Xp + Xi)
        Ntot = Sno + Snkj
        SS = 0.75 * (Xs + Xi + Xbh + Xba + Xp)
        BOD5 = 0.25 * (Ss + Xs + (1 - fp) * (Xbh + Xba))
        COD = Ss + Si + Xs + Xi + Xbh + Xba + Xp

        TSN = Sno + Snh + Snd

        EQI = (B_ss * SS + B_COD * COD + B_NKj * Snkj + B_NO * Sno + B_BOD * BOD5) * (1 / 1000) * 0.66

        SP_total = waste_sX_weight + sX_eff

        eff_component = [V, Ntot, COD, Snh, BOD5, Sno, Qw]

        return EQI, eff_component


