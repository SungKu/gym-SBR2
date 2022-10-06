import numpy as np


def sbr_reward(x_out, u_t, done, t_range, Kla, EC,EC_conc,reward_EQI_t, reward_OCI_t, reward_AE_t, reward_EC_t):
     
    so = x_out[8]
    snh = x_out[10]
    t_delta = 0.002 / 24
    So_sat = 8 ;
    # ========= EQI ==========

    V = x_out[0]
    Si = x_out[1]
    Ss = x_out[2]
    Xi = x_out[3]
    Xs = x_out[4]
    Xbh = x_out[5]
    Xba = x_out[6]
    Xp = x_out[7]
    So = x_out[8]
    Sno = x_out[9]
    Snh = x_out[10]
    Snd = x_out[11]
    Xnd = x_out[12]
    Salk = x_out[13]

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

    if Snh<4:
        penalty_Snh = 0
    else:
        penalty_Snh = -0.5

    if Ntot < 18:
        penalty_Ntot = 0
    else:
        penalty_Ntot = -0.5


    EQI2 = EQI/10 #+ penalty_Snh + penalty_Ntot
    #print("EQI2     in steps for reward: {}".format(EQI2))

    # ========= OCI ==========


    # Aeration energy (kWh / d)
    # print("t after run_step: {}".format(len(t_range_step)))

    # AE
    AE_deltaT = 1.32*sum(Kla[-len(t_range):-1])*t_delta
    AE_OCI = So_sat / ((t_range[-1] - t_range[0])*1.8 * 1000) * (AE_deltaT)
    AE_OCI_max = 1.32*(240*11)*t_delta * (So_sat / ((t_delta*11)*1.8 * 1000)) #240: Kla Max
    AE_OCI2 = AE_OCI/AE_OCI_max
    #print("AE_OCI     in steps for reward: {}".format(AE_OCI))
    #print("AE_OCI_max in steps for reward: {}".format(AE_OCI_max))
    #print("AE_OCI2     in steps for reward: {}".format(AE_OCI2))

    # EC
    EC_OCI = EC_conc*sum(EC[-len(t_range):-1])*t_delta/((t_range[-1] - t_range[0])*1000)
    EC_OCI_max = EC_conc*(0.0005*11)*t_delta/((t_delta*11)*1000) # 0.0005: EC max
    EC_OCI2 = EC_OCI/EC_OCI_max
    
    #if EC_OCI > 0:
    #    print("EC_OCI     in steps for reward: {}".format(EC_OCI))
    #    print("EC_OCI_max in steps for reward: {}".format(EC_OCI_max))
    #    print("EC_OCI2     in steps for reward: {}".format(EC_OCI2))
    
    
    #print("Kla in steps for reward: {}".format(Kla[-len(t_range):-1]))
    #print("sum(Kla) in steps for reward: {}".format(sum(Kla[-len(t_range):-1])))
    #print("AE_deltaT in reward: {}".format(AE_deltaT))
    #print("AE in reward: {}".format(AE))
    #print("EQI in reward: {}".format(EQI))

    OCI = AE_OCI + EC_OCI
    OCI2 = AE_OCI2 + EC_OCI2
    

    #if EC_OCI > 0:
        #print("EC in reward: {}".format(EC_OCI))
        #print("EQI in reward: {}".format(EQI))
        #print("OCI in reward: {}".format(OCI))

    
    reward = 1-(EQI2**2 + OCI**2)      
    #print("reward: {}".format(reward))
    reward = reward/473
    
    reward_EQI_t.append(EQI2)
    reward_OCI_t.append(OCI2)
    reward_AE_t.append(AE_OCI2)
    reward_EC_t.append(EC_OCI2)


    return reward
