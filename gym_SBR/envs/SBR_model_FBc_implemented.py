"""Author: SKHEO, KHU"""

from gym_SBR.envs import sub_phases_PID_on as cycle
import numpy as np


#class SBR_model(object):
def run(WV, IV, t_ratio, influent, DO_control_par, x0, DO_setpoints):


    """
    Ref.:
            MN Pons*, et al., Definition of a Benchmark Protocol for Sequencing Batch Reactors (B-SBR),  IFAC Proceedings Volumes
            Volume 37, Issue 3, March 2004, Pages 439-444

            * One of the developer of BSM.
    """

    """
    Basic phase sequencing:
    
            Phase No./      Feeding     Aeration    Mixing      Discharge/  Type
            length(%)                                           Wastage
            1 (4.2)         Yes         No          Yes         No          FLL/Rxn (ANX)    
            2 (8.3)         No          No          Yes         No          Rxn (ANX)
            3 (37.5)        No          Yes         Yes         No          Rxn (AER)
            4 (31.2)        No          No          Yes         No          Rxn (ANX)
            5 (2.1)         No          Yes         Yes         No          Rxn (AER)
            6 (8.3)         No          No          No          No          STL
            7 (2.1)         No          No          No          Yes         DRW
            8 (6.3)         No          Yes         No          No          IDL
    
            (ref. Pons et al. Tbl. 1) 
    """


    # Plant Config.
    WV =  WV # m^3, Working Volume
    IV = IV # m^3, Inoculum Volume

    # phase time
    t_cycle = 12 / 24  # hour -> day, 12hr
    t_phs1 = (t_cycle) * t_ratio[0]  # around 30 min
    t_phs2 = (t_cycle) * t_ratio[1]
    t_phs3 = (t_cycle) * t_ratio[2]
    t_phs4 = (t_cycle) * t_ratio[3]
    t_phs5 = (t_cycle) * t_ratio[4]
    t_phs6 = (t_cycle) * t_ratio[5]
    t_phs7 = (t_cycle) * t_ratio[6]
    t_phs8 = (t_cycle) * t_ratio[7]

    t_delta = 0.002  /24# 0.5 / 24 # min

    # flowrate
    Qin = (WV - IV)
    qin = Qin / (t_phs1)  # per day

    # solver: Runge-Kutta alg., constant integration step size: 0.002 hr

    # system stabilisation
    # stabilization time : 100 day
    # cycle : 12hr
    # num. of phases: 8


    """
     List of variables :
                0=V, 1=Si, 2=Ss, 3=Xi, 4=Xs, 5=Xbh, 6=Xba, 7=Xp, 8=So, 9=Sno, 10=Snh, 11=Snd, 12=Xnd, 13=Salk
                (ref. BSM1 report Tbl. 1)
    """

    """
     Stoichiometric parameters :
                0=Ya        1=Yh    2=fp    3=ixb   4=ixp
                (ref. BSM1 report Tbl. 2)
    """

    """
     Kinetic parameters :
                0=muhath    1=Ks    2=Koh   3=Kno     4=bh     5=etag
                6=etah      7=kh    8=Kx    9=muhata  10=Knh   11=ba
                12=Koa      13=Ka
                (ref. BSM1 report Tbl. 3) 
    """

    """
     Dissolved oxygen control parameters :
                0=Kc [h-1 (mg/L)-1]    1=taui [hr]     2=delt [hr]      3=So_set [mg/L]  4=Kla_min [hr-1] 
                5=Kla_max [hr-1]       6=DKla_max [hr-1]                7=So_low [mg/L] 8=So_high [mg/L]
                (ref. Pons et al. Tbl. 2) 
    """

    # Inflowrate - constant flow

    loading = influent
    # V,  Si, Ss, Xi, Xs, Xbh, Xba, Xp, So, Sno, Snh, Snd, Xnd, Salk
    # Buffer tank

    # Parameters

    Spar = [0.24, 0.67, 0.08, 0.08, 0.06]  # (ref. BSM1 report Tbl. 2)
    #       Ya    Yh    fp    ixb   ixp
    Kpar = [4.0, 10.0, 0.2, 0.5, 0.3, 0.8, 0.8, 3.0, 0.1, 0.5, 1.0, 0.05, 0.4, 0.05]  # (ref. BSM1 report Tbl. 3)
    #      muhath  Ks  Koh  Kno  bh   tag etah   kh   Kx muhata Knh  ba   Koa   Ka

    # initial values

    # x0 =[0, 150, 0, 0, 400, 30, 0, 3, 0, 0, 0, 5,0]
    # Si, Ss, Xi, Xs, Xbh, Xba, Xp, So, Sno, Snh, Snd, Xnd, Salk


    # result plot
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []
    x8 = []
    x9 = []
    x10 = []
    x11 = []
    x12 = []
    x13 = []
    x14 = []
    t = []

    history_eff = np.zeros((100, 14))
    history_EQ = []

    history_AE = []
    history_ME = []
    history_PE = []
    history_SP = []
    history_EC = []



    t_start = 0
    t_end = 0
    for cc in range(1) :

        phs1 = cycle.filling(WV, qin)
        DO_control_par[3] = DO_setpoints[0]
        kla0 = DO_control_par[5]

        t_start = t_end
        t_end = t_start + t_phs1
        t_save1, x_phs1, AE_1, ME_1 , kla_memory1, sp_memory1, So_memory1 = phs1.sim_rxn(t_start, t_end, t_delta, x0, Spar, Kpar, DO_control_par, loading, kla0)

        for i in range(int(len(t_save1))):
            x1.append((x_phs1[i])[0])
            x2.append((x_phs1[i])[1])
            x3.append((x_phs1[i])[2])
            x4.append((x_phs1[i])[3])
            x5.append((x_phs1[i])[4])
            x6.append((x_phs1[i])[5])
            x7.append((x_phs1[i])[6])
            x8.append((x_phs1[i])[7])
            x9.append((x_phs1[i])[8])
            x10.append((x_phs1[i])[9])
            x11.append((x_phs1[i])[10])
            x12.append((x_phs1[i])[11])
            x13.append((x_phs1[i])[12])
            x14.append((x_phs1[i])[13])
            t.append(t_save1[i])

        phs2 = cycle.rxn(WV)
        #DO_control_par[5] = 0
        DO_control_par[3] = DO_setpoints[1]

        t_start = t_end + t_delta
        t_end = t_start + t_phs2
        t_save2, x_phs2, AE_2, ME_2 , kla_memory2, sp_memory2, So_memory2 = phs2.sim_rxn(t_start, t_end, t_delta, x_phs1[-1], Spar, Kpar, DO_control_par, kla_memory1[-1])

        for i in range(int(len(t_save2))):
            x1.append((x_phs2[i])[0])
            x2.append((x_phs2[i])[1])
            x3.append((x_phs2[i])[2])
            x4.append((x_phs2[i])[3])
            x5.append((x_phs2[i])[4])
            x6.append((x_phs2[i])[5])
            x7.append((x_phs2[i])[6])
            x8.append((x_phs2[i])[7])
            x9.append((x_phs2[i])[8])
            x10.append((x_phs2[i])[9])
            x11.append((x_phs2[i])[10])
            x12.append((x_phs2[i])[11])
            x13.append((x_phs2[i])[12])
            x14.append((x_phs2[i])[13])
            t.append(t_save2[i])

        phs3 = cycle.rxn(WV)
        #DO_control_par[5] = 240
        DO_control_par[3] = DO_setpoints[2]


        t_start = t_end + t_delta
        t_end = t_start + t_phs3
        t_save3, x_phs3, AE_3, ME_3  , kla_memory3, sp_memory3, So_memory3 = phs3.sim_rxn(t_start, t_end, t_delta, x_phs2[-1], Spar, Kpar, DO_control_par, kla_memory2[-1])

        for i in range(int(len(t_save3))):
            x1.append((x_phs3[i])[0])
            x2.append((x_phs3[i])[1])
            x3.append((x_phs3[i])[2])
            x4.append((x_phs3[i])[3])
            x5.append((x_phs3[i])[4])
            x6.append((x_phs3[i])[5])
            x7.append((x_phs3[i])[6])
            x8.append((x_phs3[i])[7])
            x9.append((x_phs3[i])[8])
            x10.append((x_phs3[i])[9])
            x11.append((x_phs3[i])[10])
            x12.append((x_phs3[i])[11])
            x13.append((x_phs3[i])[12])
            x14.append((x_phs3[i])[13])
            t.append(t_save3[i])

        phs4 = cycle.rxn(WV)
        #DO_control_par[5] = 0
        DO_control_par[3] = DO_setpoints[3]


        t_start = t_end + t_delta
        t_end = t_start + t_phs4
        t_save4, x_phs4, AE_4, ME_4 , kla_memory4, sp_memory4, So_memory4 = phs4.sim_rxn(t_start, t_end, t_delta, x_phs3[-1], Spar, Kpar, DO_control_par, kla_memory3[-1])

        for i in range(int(len(t_save4))):
            x1.append((x_phs4[i])[0])
            x2.append((x_phs4[i])[1])
            x3.append((x_phs4[i])[2])
            x4.append((x_phs4[i])[3])
            x5.append((x_phs4[i])[4])
            x6.append((x_phs4[i])[5])
            x7.append((x_phs4[i])[6])
            x8.append((x_phs4[i])[7])
            x9.append((x_phs4[i])[8])
            x10.append((x_phs4[i])[9])
            x11.append((x_phs4[i])[10])
            x12.append((x_phs4[i])[11])
            x13.append((x_phs4[i])[12])
            x14.append((x_phs4[i])[13])
            t.append(t_save4[i])

        phs5 = cycle.rxn(WV)
        #DO_control_par[5] = 240
        DO_control_par[3] = DO_setpoints[4]


        t_start = t_end + t_delta
        t_end = t_start + t_phs5
        t_save5, x_phs5, AE_5, ME_5 , kla_memory5, sp_memory5, So_memory5 = phs5.sim_rxn(t_start, t_end, t_delta, x_phs4[-1], Spar, Kpar, DO_control_par, kla_memory4[-1])

        for i in range(int(len(t_save5))):
            x1.append((x_phs5[i])[0])
            x2.append((x_phs5[i])[1])
            x3.append((x_phs5[i])[2])
            x4.append((x_phs5[i])[3])
            x5.append((x_phs5[i])[4])
            x6.append((x_phs5[i])[5])
            x7.append((x_phs5[i])[6])
            x8.append((x_phs5[i])[7])
            x9.append((x_phs5[i])[8])
            x10.append((x_phs5[i])[9])
            x11.append((x_phs5[i])[10])
            x12.append((x_phs5[i])[11])
            x13.append((x_phs5[i])[12])
            x14.append((x_phs5[i])[13])
            t.append(t_save5[i])


        phs6 = cycle.settling()

        t_start = t_end + t_delta
        t_end  = t_start + t_phs6
        t_save6, Xnd, sX, Xf = phs6.sim_settling(t_start,t_end,t_delta,x_phs5[-1])


        """
         List of variables :
                    0=V, 1=Si, 2=Ss, 3=Xi, 4=Xs, 5=Xbh, 6=Xba, 7=Xp, 8=So, 9=Sno, 10=Snh, 11=Snd, 12=Xnd, 13=Salk
                    (ref. BSM1 report Tbl. 1)
        """

        for i in range(int(len(t_save6))):
            x1.append((x_phs5[-1])[0])
            x2.append((x_phs5[-1])[1])
            x3.append((x_phs5[-1])[2])
            x4.append((x_phs5[-1])[3])
            x5.append((x_phs5[-1])[4])
            x6.append((x_phs5[-1])[5])
            x7.append((x_phs5[-1])[6])
            x8.append((x_phs5[-1])[7])
            x9.append((x_phs5[-1])[8])
            x10.append((x_phs5[-1])[9])
            x11.append((x_phs5[-1])[10])
            x12.append((x_phs5[-1])[11])
            x13.append((x_phs5[-1])[12])
            x14.append((x_phs5[-1])[13])
            t.append(t_save6[i])


        # fractions: convert sludge to particular conc.
        f_xs = x5[-1]/Xf
        f_xp = x8[-1]/Xf
        f_xi = x4[-1]/Xf
        f_xbh = x6[-1]/Xf
        f_xba = x7[-1]/Xf

        # Waste sludge conc.
        w_Xs = f_xs * sX[0]
        w_Xp = f_xp  * sX[0]
        w_Xi = f_xi * sX[0]
        w_Xbh = f_xbh * sX[0]
        w_Xba = f_xba * sX[0]

        biomass_setpoint = 5400 #2700*(x_phs5[-1][3] + x_phs5[-1][4] + x_phs5[-1][5] + x_phs5[-1][6] + x_phs5[-1][7])/(x_phs5[-1][5] + x_phs5[-1][6])
        biomass_eff = sX[-1]#*(f_xbh+f_xba)
        biomass_w = sX[0]#w_Xbh + w_Xba


        #Qw = ((x6[-1] + x7[-1])*WV - biomass_setpoint*(WV-qin*t_phs1) - qin*t_phs1* biomass_eff)/(biomass_w-biomass_eff)
        Qw =  (sum(sX)*WV/10 - biomass_setpoint*(WV-qin*t_phs1) - qin*t_phs1* biomass_eff)/(biomass_w-biomass_eff)
        Qeff = qin*t_phs1 - Qw

        qeff = Qeff / t_phs7
        qw =  Qw/ t_phs7

        phs7 = cycle.drawing()
        t_start = t_end + t_delta
        t_end  = t_start + t_phs7
        t_save7, x_phs7 , PE_7, SP_7 = phs7.sim_drawing(t_start,t_end,t_delta,x_phs5[-1], sX,Xf, Qin, Qeff,Qw)



        for i in range(int(len(t_save7))):
            x1.append((x_phs7[0]))
            x2.append((x_phs7[1]))
            x3.append((x_phs7[2]))
            x4.append((x_phs7[3]))
            x5.append((x_phs7[4]))
            x6.append((x_phs7[5]))
            x7.append((x_phs7[6]))
            x8.append((x_phs7[7]))
            x9.append((x_phs7[8]))
            x10.append((x_phs7[9]))
            x11.append((x_phs7[10]))
            x12.append((x_phs7[11]))
            x13.append((x_phs7[12]))
            x14.append((x_phs7[13]))
            t.append(t_save7[i])

        # effluent information
        eq, eff = phs7.cal_eq(x_phs7,sX, Xf, x_phs5[-1], Spar, Qeff)

        history_eff[cc] = eff
        history_EQ.append(eq)

        phs8 = cycle.rxn(WV)
        #DO_control_par[5] = 240
        DO_control_par[3] =DO_setpoints[7]


        t_start = t_end + t_delta
        t_end = t_start + t_phs8
        t_save8, x_phs8, AE_8, ME_8, kla_memory8 , sp_memory8, So_memory8 = phs8.sim_rxn(t_start, t_end, t_delta, x_phs7, Spar, Kpar, DO_control_par, kla_memory5[-1])

        for i in range(int(len(t_save8))):
            x1.append((x_phs8[i])[0])
            x2.append((x_phs8[i])[1])
            x3.append((x_phs8[i])[2])
            x4.append((x_phs8[i])[3])
            x5.append((x_phs8[i])[4])
            x6.append((x_phs8[i])[5])
            x7.append((x_phs8[i])[6])
            x8.append((x_phs8[i])[7])
            x9.append((x_phs8[i])[8])
            x10.append((x_phs8[i])[9])
            x11.append((x_phs8[i])[10])
            x12.append((x_phs8[i])[11])
            x13.append((x_phs8[i])[12])
            x14.append((x_phs8[i])[13])
            t.append(t_save8[i])



        x0 = [x1[-1], x2[-1], x3[-1], x4[-1], x5[-1], x6[-1], x7[-1], x8[-1], x9[-1], x10[-1], x11[-1], x12[-1], x13[-1], x14[-1]] # 공정의 마지막 값, 다음 cycle의 initial state

        x = np.vstack([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14])





    return t,x,x0, kla_memory1, So_memory1, sp_memory1, t_save1, kla_memory2, So_memory2, sp_memory2, t_save2, kla_memory3, So_memory3, sp_memory3, t_save3, kla_memory4, So_memory4, sp_memory4, t_save4, kla_memory5, So_memory5, sp_memory5, t_save5, kla_memory8, So_memory8, sp_memory8, t_save8, Qeff,Qw
