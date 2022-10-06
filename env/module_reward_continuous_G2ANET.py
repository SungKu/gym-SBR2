import numpy as np


def sbr_reward(x_out, u_t, done, eff):
    ss = x_out[2]
    so = x_out[8]
    sno = x_out[9]
    snh = x_out[10]
    t_delta = 0.002 / 24


    # ========= OCI ==========
    dt = t_delta

    T = 0.5  # 12hrs, 0.5 day


    if ss < 0:
        r_ec = 1
    else:
        r_ec = -(ss - 0) / (10 - 0) + 1

    if so < 1.5:
        r_e = 0
    else:
        r_e = -(1 / (8 - 1.5)) * (so - 8) + 0

    if sno < 4:
        r_sno = 1
    else:
        r_sno = -(sno - 4) / (20 - 4) + 1

    if snh < 4:
        r_snh = 1
    else:
        r_snh = -(snh - 4) / (20 - 4) + 1


    w_ec, w_e, w_sno, w_snh = 1, 1.5, 2, 2

    reward = w_ec*r_ec + w_e*r_e  + w_sno*r_sno +  w_snh*r_snh
    reward = reward/10
    # print("reward in reward function:",reward)

    return reward
