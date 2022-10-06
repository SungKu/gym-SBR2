""" SungKu Heo, KHU """

# Assign the, Volume, Time length of each sequence, Influent,  Controller, etc to 'SBR_model'.

from gym_SBR.envs import buffer_tank2 as buffer_tank
from gym_SBR.envs import SBR_model_batchPID_fbPID as SBR
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import component_figure

# Memory

memory_switch= []
memory_influent_mixed = []
memory_influent_var= []

memory_component_state = []
memory_time = []

# Plant Config.
WV = 1.32  # m^3, Working Volume
IV = 0.66  # m^3, Inoculum Volume

t_ratio = [4.2/100, 8.3/100, 37.5/100, 31.2/100, 2.1/100, 8.3/100, 2.1/100, 6.3/100]


# Dissolved oxygen control parameters
DO_control_par = [21.6, 0.05/24, 0.02/24, 8, 0, 240, 12, 2, 5]
#DO_control_par = [0.9, 0.05, 0.02, 8, 0, 10, 0.5, 2, 5]
# Kc, taui, delt, So_set, Kla_min, Kla_max, DKla_max So_low, So_high

#Initial state from stablization,
x0 = [IV, 30.0, 0.5601630529230822, 1762.3890076468106, 30.97046860269441, 2628.6551849696393, 188.71238190722482,
      780.479571994941, 6.83620016588177, 14.575400491942467, 0.00872090237410032, 0.36940333660700486,
      1.896711744868243, 3.705237172170034]

switch, influent_mixed , influent_var = buffer_tank.influent.buffer_tank(0,12)
influent_mixed[0] =  31.4285

memory_switch.append(switch)
memory_influent_mixed.append(influent_mixed)
memory_influent_var.append(influent_var)

DO_setpoints = [0,0,2,0,2,0,0,2]

t, x, x_last= SBR.run(WV, IV, t_ratio, influent_mixed, DO_control_par,x0, DO_setpoints)

memory_component_state.append(x)
memory_time.append(t)


component_figure.figure(t,x)
