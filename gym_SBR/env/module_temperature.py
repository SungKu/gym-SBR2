import numpy as np

def DO_set(T_as):

    # Oxygen concentration at saturation
    # is function of Temp.

    #T_as = 15
    T_k = T_as + 273.15
    T_ast = T_k / 100

    A = -66.7354
    B = 87.4755
    C = 24.4526

    f_Tk = 56.12 * np.exp(A + B / T_ast + C * np.log(T_ast))

    So_sat = 0.9997743214 * (8 / 10.5) * 6791.5 * f_Tk

    return So_sat
