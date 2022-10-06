"""Author: SKHEO, KHU"""

from gym_SBR.envs import sub_phases_continuous as cycle
import numpy as np


#class SBR_model(object):
def run(WV, IV, t_ratio, influent, DO_control_par, x0, DO_setpoints, kla0):


    loading = influent
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


    # Parameters

    Spar = [0.24, 0.67, 0.08, 0.08, 0.06]  # (ref. BSM1 report Tbl. 2)
    #       Ya    Yh    fp    ixb   ixp
    Kpar = [4.0, 10.0, 0.2, 0.5, 0.3, 0.8, 0.8, 3.0, 0.1, 0.5, 1.0, 0.05, 0.4, 0.05]  # (ref. BSM1 report Tbl. 3)
    #      muhath  Ks  Koh  Kno  bh   tag etah   kh   Kx muhata Knh  ba   Koa   Ka


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

    t_start = 0
    t_end = 0
    for cc in range(1) :

        phs1 = cycle.filling(WV, qin)
        DO_control_par[3] = DO_setpoints[0]
        #kla0 = DO_control_par[5]

        t_start = t_end
        t_end = t_start + t_phs1
        t_save1, x_phs1, AE_1, ME_1 , kla1, sp_memory1, So_memory1 = phs1.sim_rxn(t_start, t_end, t_delta, x0, Spar, Kpar, DO_control_par, loading, kla0)

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
        t_save2, x_phs2, AE_2, ME_2 , kla2, sp_memory2, So_memory2 = phs2.sim_rxn(t_start, t_end, t_delta, x_phs1[-1], Spar, Kpar, DO_control_par, kla1[-1])

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
        t_save3, x_phs3, AE_3, ME_3  , kla3, sp_memory3, So_memory3 = phs3.sim_rxn(t_start, t_end, t_delta, x_phs2[-1], Spar, Kpar, DO_control_par, kla2[-1])

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
        t_save4, x_phs4, AE_4, ME_4 , kla4, sp_memory4, So_memory4 = phs4.sim_rxn(t_start, t_end, t_delta, x_phs3[-1], Spar, Kpar, DO_control_par, kla3[-1])

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
        t_save5, x_phs5, AE_5, ME_5 , kla5, sp_memory5, So_memory5 = phs5.sim_rxn(t_start, t_end, t_delta, x_phs4[-1], Spar, Kpar, DO_control_par, kla4[-1])

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



        biomass_setpoint = 2700

        Qeff =0.66# WV-IV

        qeff = Qeff / t_phs7


        phs7 = cycle.drawing()
        t_start = t_end + t_delta
        t_end  = t_start + t_phs7
        t_save7, x_phs7 , Qw,PE_7, SP_7, EQI, eff = phs7.sim_drawing(t_start,t_end,t_delta,x_phs5[-1], sX,Xf, Qeff,biomass_setpoint)






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


        phs8 = cycle.rxn(WV)
        #DO_control_par[5] = 240
        DO_control_par[3] =DO_setpoints[7]


        t_start = t_end + t_delta
        t_end = t_start + t_phs8
        t_save8, x_phs8, AE_8, ME_8, kla8 , sp_memory8, So_memory8 = phs8.sim_rxn(t_start, t_end, t_delta, x_phs7, Spar, Kpar, DO_control_par, kla5[-1])

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





    return t,x,x0, sp_memory3, So_memory3, t_save3, sp_memory5, So_memory5, t_save5, sp_memory8, So_memory8, t_save8,Qeff, eff,Qw,kla3,kla5, kla8, EQI
