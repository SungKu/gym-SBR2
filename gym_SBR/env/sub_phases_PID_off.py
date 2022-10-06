import scipy.integrate as integrate
import numpy as np
import copy
import matplotlib.pyplot as plt


"""
Ref.:
        MN Pons*, et al., Definition of a Benchmark Protocol for Sequencing Batch Reactors (B-SBR),  IFAC Proceedings Volumes
        Volume 37, Issue 3, March 2004, Pages 439-444

        * One of the developer of BSM.
"""

"""
 List of variables :
            0=Si, 1=Ss, 2=Xi, 3=Xs, 4=Xbh, 5=Xba, 6=Xp, 
            7=So, 8=Sno, 9=Snh, 10=Snd, 11=Xnd, 12=Salk
            (ref. BSM1 report Tbl. 1)
"""

"""
 Stoichiometric parameters :
            0=Ya  1=Yh    2=fp    3=ixb   4=ixp
            (ref. BSM1 report Tbl. 2)
"""

"""
 Kinetic parameters :
            0=muhath    1=Ks    2=Koh   3=Kno     4=bh     5=etag
            6=etah      7=kh    8=Kx    9=muhata  10=Knh   11=ba
            12=Koa      13=ka
            (ref. BSM1 report Tbl. 3) 
"""

"""
 Dissolved oxygen control parameters :
            0=Kc [h-1 (mg/L)-1]    1=taui [hr]     2=delt [hr]      3=So_set [mg/L]  4=Kla_min [hr-1] 
            5=Kla_max [hr-1]       6=DKla_max [hr-1]                7=So_low [mg/L] 8=So_high [mg/L]
            (ref. Pons et al. Tbl. 2) 
"""



class filling(object):
    def __init__(self,WV, qin):
        self.__WV = WV
        self.__q = qin


    def dxdt(self, x, t, Spar, Kpar,DO_control_par,Kla, loading):
        # Process 1
        rho1 = Kpar[0] * (x[2] / (Kpar[1] + x[2])) * (x[8] / (Kpar[2] + x[8])) * x[5]

        # Process 2
        rho2 = Kpar[0] * (x[2] / (Kpar[1] + x[2])) * (Kpar[2] / (x[8] + Kpar[2])) * (x[9] / (Kpar[3] + x[9])) * Kpar[
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
        rho7 = Kpar[7] * ((x[4] / x[5]) / (Kpar[8] + (x[4] / x[5]))) * ((x[8] / (Kpar[2] + x[8])) + Kpar[6] * (Kpar[2] / (x[8] + Kpar[2])) * (x[9] / (Kpar[3] + x[9])))*x[5]

        # Process 8
        rho8 = (x[12] / x[4])*rho7

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
        dxdt[1] = r1 + (loading[0]/x[0])*(loading[1] - x[1])
        # 2 = Ss,
        dxdt[2] = r2 + (loading[0]/x[0])*(loading[2] - x[2])
        # 3 = Xi,
        dxdt[3] = r3 + (loading[0]/x[0])*(loading[3] - x[3])
        # 4 = Xs,
        dxdt[4] = r4 + (loading[0]/x[0])*(loading[4] - x[4])
        # 5 = Xbh,
        dxdt[5] = r5 + (loading[0]/x[0])*(loading[5] - x[5])
        # 6 = Xba,
        dxdt[6] = r6 + (loading[0]/x[0])*(loading[6] - x[6])
        # 7 = Xp,
        dxdt[7] = r7 + (loading[0]/x[0])*(loading[7] - x[7])
        # 8 = So,
        dxdt[8] = r8 + (loading[0]/x[0])*(loading[8] - x[8])
        # 9 = Sno,
        dxdt[9] = r9 + (loading[0]/x[0])*(loading[9] - x[9])
        # 10 = Snh,
        dxdt[10] = r10 + (loading[0]/x[0])*(loading[10] - x[10])
        # 11 = Snd,
        dxdt[11] = r11 + (loading[0]/x[0])*(loading[11] - x[11])
        # 12 = Xnd,
        dxdt[12] = r12 + (loading[0]/x[0])*(loading[12] - x[12])
        # 13 = Salk
        dxdt[13] = r13 + (loading[0]/x[0])*(loading[13] - x[13])

        return dxdt

    def sim_rxn(self,t_start, t_end,t_delta, x, Spar,Kpar,DO_control_par,loading, kla):

        # Fix the Kla, detach the control
        #kla = DO_control_par[5]

        # derivate time(day)
        dt = DO_control_par[2]

        t_save = np.linspace(t_start,t_end,(t_end - t_start)/t_delta)
        t_save2 = np.linspace(t_start, t_end, (t_end - t_start) / (t_delta*10))

        # Memory variables
        Kla_memory = [] #np.zeros(1+(len(t_save2)-1)*(int(dt/t_delta)-1)) # Kla, Manipulate variable
        So_memory = [] # So, Controlled variable
        sp_memory = []  # set-points
        x_memory =  [] # state variables
        t_memory =  [] # time variables

        #initial value
        Kla_memory.append(kla)
        So_memory.append(x[8])
        sp_memory.append(DO_control_par[3])
        x_memory.append(x) # state at t_start
        t_memory.append(t_save2[0]) # time at t_start


        for i in range(len(t_save2)-1) :

            t_s_flag = t_save2[i]
            t_e_flag = t_save2[i+1]

            t_range = np.linspace(t_s_flag,t_e_flag,(t_e_flag-t_s_flag)/t_delta)

            soln = integrate.odeint(self.dxdt, x, t_range, args=( Spar, Kpar,DO_control_par,kla,loading,))#,full_output=0)#,rtol=1.49012e-2)

            for ii in range(len(t_range)-1):
                x_memory.append(soln[ii+1])
                t_memory.append(t_range[ii+1])
                So_memory.append(soln[ii+1][8])
                Kla_memory.append(kla)
                sp_memory.append(DO_control_par[3])

            x = soln[-1]

        AE = 0
        ME = 0

        return  t_memory, x_memory, AE, ME, Kla_memory, So_memory, sp_memory


class rxn(object):
    def __init__(self,WV):
        self.__WV = WV

    def dxdt(self, x, t, Spar, Kpar,DO_control_par,Kla):
        # Process 1
        rho1 = Kpar[0] * (x[2] / (Kpar[1] + x[2])) * (x[8] / (Kpar[2] + x[8])) * x[5]

        # Process 2
        rho2 = Kpar[0] * (x[2] / (Kpar[1] + x[2])) * (Kpar[2] / (x[8] + Kpar[2])) * (x[9] / (Kpar[3] + x[9])) * Kpar[
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
                (x[8] / (Kpar[2] + x[8])) + Kpar[6] * (Kpar[2] / (x[8] + Kpar[2])) * (x[9] / (Kpar[3] + x[9]))) * x[5]

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

    def sim_rxn(self, t_start, t_end,t_delta, x, Spar, Kpar, DO_control_par, kla):

        # Fix the Kla, detach the control
        kla = DO_control_par[5]

        # derivate time(day)
        dt = DO_control_par[2]

        t_save = np.linspace(t_start,t_end,(t_end - t_start)/t_delta)
        t_save2 = np.linspace(t_start, t_end, (t_end - t_start) / (t_delta*10))

        # Memory variables
        Kla_memory = [] #np.zeros(1+(len(t_save2)-1)*(int(dt/t_delta)-1)) # Kla, Manipulate variable
        So_memory = [] # So, Controlled variable
        sp_memory = [] # set-points
        x_memory =  [] # state variables
        t_memory =  [] # time variables


        #initial value

        Kla_memory.append(kla)
        So_memory.append(x[8])
        x_memory.append(x) # state at t_start
        t_memory.append(t_save2[0]) # time at t_start
        sp_memory.append(DO_control_par[3])

        for i in range(len(t_save2)-1) :

            t_s_flag = t_save2[i]
            t_e_flag = t_save2[i+1]

            t_range = np.linspace(t_s_flag,t_e_flag,(t_e_flag-t_s_flag)/t_delta)


            soln = integrate.odeint(self.dxdt, x, t_range, args=( Spar, Kpar,DO_control_par,kla,))#,full_output=0)#,rtol=1.49012e-2)


            for ii in range(len(t_range)-1):
                x_memory.append(soln[ii+1])
                t_memory.append(t_range[ii+1])
                So_memory.append(soln[ii+1][8])
                Kla_memory.append(kla)
                sp_memory.append(DO_control_par[3])




            x = soln[-1]

        AE = 0
        ME = 0



        return  t_memory, x_memory, AE, ME , Kla_memory, So_memory, sp_memory




class settling(object):


    def dXnddt(self, Xnd, t, z, Xf):

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

    def dsXdt(self, sX, t, z, Xf):

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
        dsXdt[9] = (0- J_sX10) / z

        return dsXdt

    def sim_settling(self, t_start, t_end, t_delta, x):

        t_save = np.linspace(t_start, t_end, (t_end - t_start) / t_delta)

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

        dXnddt = integrate.odeint(self.dXnddt, Xnd, t_save, args=(z,Xf,))
        dsXdt = integrate.odeint(self.dsXdt, sX, t_save, args=(z, Xf,))

        Xnd = dXnddt[-1]
        sX = dsXdt[-1]

        return t_save, Xnd, sX, Xf



class drawing(object):


    def sim_drawing(self, t_start, t_end, t_delta, x,sX, Xf , Qin, Qeff, Qw):

        t_save = np.linspace(t_start, t_end, (t_end - t_start) / t_delta)

        """
         List of variables :
                    0=V, 1=Si, 2=Ss, 3=Xi, 4=Xs, 5=Xbh, 6=Xba, 7=Xp, 8=So, 9=Sno, 10=Snh, 11=Snd, 12=Xnd, 13=Salk
                    (ref. BSM1 report Tbl. 1)
        """

        init_V = x[0]


        V = init_V - Qeff - Qw
        sX2 = (sum(sX)*init_V/10 - Qw*sX[0] - Qeff*sX[-1])/V

        f_xs = 0.75*x[4] / Xf
        f_xp = 0.75*x[7] / Xf
        f_xi = 0.75*x[3] / Xf
        f_xbh = 0.75*x[5] / Xf
        f_xba = 0.75*x[6] / Xf

        x_phs7 = copy.deepcopy(x[:])

        x_phs7[0]= V
        x_phs7[4] = f_xs *sX2
        x_phs7[7] = f_xp *sX2
        x_phs7[3] = f_xi *sX2
        x_phs7[5] = f_xbh *sX2
        x_phs7[6] = f_xba *sX2

        #soluable component

        # particulate component


        #PE: Pumping energy, (kWh/d)

        PE = 0.05*abs(Qw)

        # sludge production, (Kg/d)
        SP = sX2*V + Qw*sX[0] + Qeff*sX[-1]





        return t_save, x_phs7, PE, SP


    """
     Stoichiometric parameters :
                0=Ya  1=Yh    2=fp    3=ixb   4=ixp
                (ref. BSM1 report Tbl. 2)
    """

    def cal_eq(self, x, sX, Xf,x_phs5, Spar, Qeff):


        sX2 = sX[-1]

        f_xs = 0.75*x_phs5[4] / Xf
        f_xp = 0.75*x_phs5[7] / Xf
        f_xi = 0.75*x_phs5[3] / Xf
        f_xbh = 0.75*x_phs5[5] / Xf
        f_xba = 0.75*x_phs5[6] / Xf
        f_xnd = 0.75*x[12] / Xf

        x_phs7 = copy.deepcopy(x[:])


        Xs = f_xs *sX2
        Xp = f_xp *sX2
        Xi = f_xi *sX2
        Xbh = f_xbh *sX2
        Xba = f_xba *sX2
        Xnd = f_xnd * sX2
        """
           List of variables :
                      0=Si, 1=Ss, 2=Xi, 3=Xs, 4=Xbh, 5=Xba, 6=Xp, 
                      7=So, 8=Sno, 9=Snh, 10=Snd, 11=Xnd, 12=Salk
                      (ref. BSM1 report Tbl. 1)
          """
        q_eff = Qeff
        Snkj= x_phs7[10] + x_phs7[11] + x_phs7[12] + Spar[3]*(Xbh + Xba) + Spar[4]*(Xp + Xi)
        SSe = 0.75*(Xs + Xp + Xi + Xbh + Xba)
        BOD5 = 0.25*(x_phs7[2] + x_phs7[4] + (1 - Spar[2])*(Xbh + Xba))
        CODe =  x_phs7[2] + x_phs7[1] + Xs + Xi + Xbh + Xba + Xp

        Bss = 2
        Bcod = 1
        Bnkj = 30
        Bno = 10
        Bbod5 = 2

        eq = (Bss*SSe + Bcod*CODe + Bnkj*Snkj+ Bno*x[9] + Bbod5*BOD5)*q_eff




        eff = [q_eff, x[1],x[2],Xi,Xs,Xbh,Xba,Xp,x[8],x[9],x[10],x[11],Xnd,x[13]]

        return eq, eff


