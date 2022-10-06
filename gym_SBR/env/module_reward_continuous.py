import numpy


def sbr_reward(So_sat, Kla, batch_type, Qin, Qw,  eff):

    t_delta = 0.002/24


    #========= OCI ==========
    dt =  t_delta

    T = 0.5 # 12hrs, 0.5 day


    #Mechanical Eergy (kWh / d)
    # Assume: anerobic phase에서 mixing
    #ME_2 = 0.005*1.32*24
    #ME_4 = 0.005*1.32*24
    #ME = ME_2 + ME_4



    if batch_type == 0 :
        # Pumping energy (kWh / d)
        PE = 0.004*Qin
        # Aeration energy (kWh / d)
        AE_deltaT = 1.32 * Kla[-1] * t_delta

        r_Snh = 0

    if batch_type == 1:
        # Aeration energy (kWh / d)
        AE_deltaT = 1.32*Kla[-1]*t_delta
        #sum(kla_memory3)*t_delta/(len(kla_memory3)*t_delta)
        PE = 0

        r_Snh = 0

    if batch_type == 2:
        Q_eff = eff[0]
        Snh = eff[3]

        # Pumping energy (kWh / d)
        PE = ( 0.05* Qw + 0.004*Q_eff) # SBR에는 내부 외부 반송 없음
        # Aeration energy (kWh / d)
        AE_deltaT = 1.32*sum(Kla)*t_delta

        if Snh < 4:
            r_Snh = 0
        else:
            r_Snh = -246


    AE = So_sat / (1.8 * 1000) * (AE_deltaT)

    OCI = AE + PE  #+ ME #+ 5*SP

    #=============================

    r_OCI = 0.5 - OCI

    reward  =  r_OCI + r_Snh


    return reward
