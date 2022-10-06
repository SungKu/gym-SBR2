import numpy as np



def sbr_reward( x_out, u_t, done, eff):
    so = x_out[8]
    snh = x_out[10]
    t_delta = 0.002 / 24
    

    # ========= OCI ==========
    dt = t_delta

    T = 0.5  # 12hrs, 0.5 day

    # Mechanical Eergy (kWh / d)
    # Assume: anerobic phase에서 mixing
    # ME_2 = 0.005*1.32*24
    # ME_4 = 0.005*1.32*24
    # ME = ME_2 + ME_4

    if done:  # Settling, Drawing, idle phases
       print(done )
       #r_e = 5*(x_traj[0,8] - x_traj[-1,8])



    else:  # Reaction phases

        # Aeration energy (kWh / d)
        
        if So <1.5:
            r_e = -100
        elif 2.5<So<3.5:
            r_e = 0
        elif 3.5 <= So < 5:
            r_e = -10
        elif 5 <= So :
            r_e = -50
        else:
            r_e = 10
        
        """r_e = -1*(so-1)*(so-3.5) /(228)
        
        if snh<4:
            r_snh = 0
        else:    
            r_snh = -(snh - 4)/(228*16)
         """



    # AE = So_sat / (1.8 * 1000) * (AE_deltaT)

    # OCI = AE # + PE  #+ ME #+ 5*SP

    # =============================

    # r_OCI = 0.5 - OCI

    reward = r_snh +r_e # r_Snh + r_e + r_n  + r_COD # + r_OCI
    #print("reward in reward function:",reward)


    return reward
