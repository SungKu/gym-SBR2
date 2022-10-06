import scipy.integrate as integrate
import numpy as np
import copy
import matplotlib.pyplot as plt
import math

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

        # derivate time(day)
        dt = DO_control_par[2]

        t_save = np.linspace(t_start,t_end,(t_end - t_start)/t_delta)
        t_save2 = np.linspace(t_start, t_end, (t_end - t_start) / (t_delta*10))

        # Memory variables
        Kla_memory = [] #np.zeros(1+(len(t_save2)-1)*(int(dt/t_delta)-1)) # Kla, Manipulate variable
        So_memory = [] # So, Controlled variable
        x_memory =  [] # state variables
        t_memory =  [] # time variables
        sp_memory = []

        # Set-point
        sp = np.zeros(len(t_save2)-1) # set-points
        sp[:] = DO_control_par[3]

        dcv =np.zeros(len(t_save2)-1) # derivate of controlled variable
        ie = np.zeros(len(t_save2)-1)  # integral of the error
        e = np.zeros(len(t_save2)-1)
        So = np.zeros(len(t_save2)-1)
        Kla = np.zeros(len(t_save2)-1)
        P =np.zeros(len(t_save2)-1)   # proportional
        I = np.zeros(len(t_save2)-1) # integral
        D =np.zeros(len(t_save2)-1)  # derivative


        #Kla = DO_control_par[5] # initial value of Kla from previous phases

        # PID parameters
        Kc = DO_control_par[0]
        tauI = DO_control_par[1]
        tauD = DO_control_par[9]

        # PID tuning

        #initial value
        So[0] = x[8]
        Kla[0] = kla
        Kla_memory.append(kla)
        So_memory.append(x[8])
        x_memory.append(x) # state at t_start
        t_memory.append(t_save2[0]) # time at t_start
        sp_memory.append(sp[0]) # set-point at t_start


        for i in range(len(t_save2)-1) :

            t_s_flag = t_save2[i]
            t_e_flag = t_save2[i+1]

            t_range = np.linspace(t_s_flag,t_e_flag,(t_e_flag-t_s_flag)/t_delta)

            e[i] = sp[i] - So[i]

            if i >= 1: # calcuate starting on second cycle
                dcv[i] = (So[i] - So[i - 1]) / dt
                ie[i] = ie[i - 1] + e[i] * dt

            P[i] = Kc * e[i]
            I[i] = Kc / tauI * ie[i]
            D[i] = Kc * tauD * dcv[i]

            Kla[i] = P[i] + I[i] + D[i] + Kla[0] # Kla_memory[0] : Bias

            if Kla[i] >  DO_control_par[5] : # check upper limit
                Kla[i] = DO_control_par[5]
                ie[i] = ie[i] - e[i] * dt # anti-reset windup
            if Kla[i] <  DO_control_par[4]: # check lower limit
                Kla[i] = DO_control_par[4]
                ie[i] = ie[i] - e[i] * dt # anti-reset windup

            soln = integrate.odeint(self.dxdt, x, t_range, args=( Spar, Kpar,DO_control_par,Kla[i],loading,))#,full_output=0)#,rtol=1.49012e-2)


            for ii in range(len(t_range)-1):
                x_memory.append(soln[ii+1])
                t_memory.append(t_range[ii+1])
                So_memory.append(soln[ii+1][8])
                Kla_memory.append(Kla[i])
                sp_memory.append(sp[i])

            if i < len(t_save2)-2:
                So[i+1] =  soln[-1][8]

            x = soln[-1]

        AE = 0
        ME = 0


        return  t_memory, x_memory, AE, ME, Kla, sp_memory, So_memory


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

        # derivate time(day)
        dt = DO_control_par[2]

        t_save = np.linspace(t_start,t_end,(t_end - t_start)/t_delta)
        t_save2 = np.linspace(t_start, t_end, (t_end - t_start) / (t_delta*10))

        # Memory variables
        Kla_memory = [] #np.zeros(1+(len(t_save2)-1)*(int(dt/t_delta)-1)) # Kla, Manipulate variable
        So_memory = [] # So, Controlled variable
        x_memory =  [] # state variables
        t_memory =  [] # time variables
        sp_memory = []

        # Set-point
        sp = np.zeros(len(t_save2)-1) # set-points
        sp[:] = DO_control_par[3]

        dcv =np.zeros(len(t_save2)-1) # derivate of controlled variable
        ie = np.zeros(len(t_save2)-1)  # integral of the error
        e = np.zeros(len(t_save2)-1)
        So = np.zeros(len(t_save2)-1)
        Kla = np.zeros(len(t_save2)-1)
        P =np.zeros(len(t_save2)-1)   # proportional
        I = np.zeros(len(t_save2)-1) # integral
        D =np.zeros(len(t_save2)-1)  # derivative


        #Kla = DO_control_par[5] # initial value of Kla from previous phases

        # PID parameters
        Kc = DO_control_par[0]
        tauI = DO_control_par[1]
        tauD = DO_control_par[9]

        # PID tuning

        #initial value
        So[0] = x[8]
        Kla[0] = kla
        Kla_memory.append(kla)
        So_memory.append(x[8])
        x_memory.append(x) # state at t_start
        t_memory.append(t_save2[0]) # time at t_start
        sp_memory.append(sp[0]) # set-point at t_start


        for i in range(len(t_save2)-1) :

            t_s_flag = t_save2[i]
            t_e_flag = t_save2[i+1]

            t_range = np.linspace(t_s_flag,t_e_flag,(t_e_flag-t_s_flag)/t_delta)

            e[i] = sp[i] - So[i]

            if i >= 1: # calcuate starting on second cycle
                dcv[i] = (So[i] - So[i - 1]) / dt
                ie[i] = ie[i - 1] + e[i] * dt

            P[i] = Kc * e[i]
            I[i] = Kc / tauI * ie[i]
            D[i] = Kc * tauD * dcv[i]

            Kla[i] = P[i] + I[i] + D[i] + Kla[0] # Kla_memory[0] : Bias

            if Kla[i] >  DO_control_par[5] : # check upper limit
                Kla[i] = DO_control_par[5]
                ie[i] = ie[i] - e[i] * dt # anti-reset windup
            if Kla[i] <  DO_control_par[4]: # check lower limit
                Kla[i] = DO_control_par[4]
                ie[i] = ie[i] - e[i] * dt # anti-reset windup

            soln = integrate.odeint(self.dxdt, x, t_range, args=( Spar, Kpar,DO_control_par,Kla[i],))#,full_output=0)#,rtol=1.49012e-2)


            for ii in range(len(t_range)-1):
                x_memory.append(soln[ii+1])
                t_memory.append(t_range[ii+1])
                So_memory.append(soln[ii+1][8])
                Kla_memory.append(Kla[i])
                sp_memory.append(sp[i])

            if i < len(t_save2)-2:
                So[i+1] =  soln[-1][8]

            x = soln[-1]

        AE = 0
        ME = 0



        return  t_memory, x_memory, AE, ME , Kla, sp_memory, So_memory


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

    def sim_drawing(self, t_start, t_end, t_delta, x,sX, Xf , Qeff, biomass_setpoint):

        t_save = np.linspace(t_start, t_end, (t_end - t_start) / t_delta)

        init_V = x[0]
        layer_volume = init_V/10
        residual_V = init_V-Qeff

        # 1. effluent drawing(유출수 방출)

        #방류수로 방출될 layer 갯수 선정
        m = int(math.ceil(round(Qeff/(layer_volume ))))

        #effluent로 방출되는 슬러지
        sX_eff = sum(sX[-m:-1]*layer_volume)

        X_eff = copy.deepcopy(x[:])

        X_eff[0]= Qeff
        X_eff[4] = X_eff[4]*(1/0.75)*sX_eff/Xf
        X_eff[7] = X_eff[7]*(1/0.75)*sX_eff/Xf
        X_eff[3] = X_eff[3]*(1/0.75)*sX_eff/Xf
        X_eff[5] = X_eff[5]*(1/0.75)*sX_eff/Xf
        X_eff[6] = X_eff[6]*(1/0.75)*sX_eff/Xf




        #방류수로 방출되고 남은 sX와 레이어 (10-m)
        residual_sX = sX[0:10-m]
        residual_sX_weight_inLayer = layer_volume * residual_sX
        residual_sX_weight = sum(residual_sX_weight_inLayer)

        desire_sX_weight = biomass_setpoint * (residual_V)
        waste_sX_weight = residual_sX_weight - desire_sX_weight


        for i in range(10-m):

            m_q = i

            residual_sX_weight_by_discharging = waste_sX_weight - residual_sX_weight_inLayer[i]

            if residual_sX_weight_by_discharging>0:
                waste_sX_weight = residual_sX_weight_by_discharging
                residual_sX[i] = 0
                residual_sX_weight_inLayer[i] = 0
                residual_V -= layer_volume

            else :
                Qw = waste_sX_weight/(residual_sX[i]-biomass_setpoint)

                residual_sX_weight_inLayer[i] = residual_sX_weight_inLayer[i] - Qw*residual_sX[i]
                residual_V -=Qw
                residual_sX[i] = residual_sX_weight_inLayer[i]/(layer_volume-Qw)

                break


        sX2 = (sum(residual_sX_weight_inLayer))/residual_V

        x_phs7 = copy.deepcopy(x[:])

        x_phs7[0]= residual_V
        x_phs7[4] = x[4]*(1/0.75)*sX2/Xf
        x_phs7[7] = x[7]*(1/0.75)*sX2/Xf
        x_phs7[3] = x[3]*(1/0.75)*sX2/Xf
        x_phs7[5] = x[5]*(1/0.75)*sX2/Xf
        x_phs7[6] = x[6]*(1/0.75)*sX2/Xf

        #soluable component

        # particulate component


        #PE: Pumping energy, (kWh/d)

        PE = 0.05*abs(Qw)

        # sludge production, (Kg/d)
        SP = sX_eff*Qeff

        EQI, eff_component =  self.cal_eq(sX_eff, X_eff, waste_sX_weight)

        return t_save, x_phs7, Qw, PE, SP, EQI, eff_component



    def cal_eq(self,sX_eff, x_eff,waste_sX_weight ):

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


        SP_total = waste_sX_weight+sX_eff


        eff_component = [0.66, Ntot, COD, Snh, BOD5, Sno]

        return EQI, eff_component



