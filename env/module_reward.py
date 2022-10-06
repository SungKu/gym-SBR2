import numpy


def sbr_reward(DO_control_par, kla_memory3, kla_memory5, kla_memory8,Qw, EQI,Qin, Q_eff, Snh,DO_setpoints):

    t_delta = 0.002/24

    #========= OCI ==========
    dt =  t_delta
    So_sat = DO_control_par[10]
    T = 0.5 # 12hrs, 0.5 day


    #Mechanical Eergy (kWh / d)
    # Assume: anerobic phase에서 mixing

    ME_2 = 0.005*1.32*24
    ME_4 = 0.005*1.32*24

    ME = ME_2 + ME_4

    # Aeration energy (kWh / d)
    AE_3 = 1.32* sum(kla_memory3)*t_delta/(len(kla_memory3)*t_delta)
    AE_5 = 1.32* sum(kla_memory5)*t_delta/(len(kla_memory5)*t_delta)
    AE_8 = (1.32-Qw)* sum(kla_memory8)*t_delta/(len(kla_memory8)*t_delta)

    AE = So_sat/(1.8*1000)*(AE_3+ AE_5 +AE_8)

    # Pumping energy (kWh / d)

    PE = ( 0.004*Qin + 0.05* Qw + 0.004*Q_eff) # SBR에는 내부 외부 반송 없음

    OCI = AE + PE + ME #+ 5*SP

    #=============================

    if Snh< 4:
        r_Snh = 0
    else:
        r_Snh = -20

    r_OCI = 5 - OCI

    reward  =  r_OCI + r_Snh
    





    return reward, OCI





