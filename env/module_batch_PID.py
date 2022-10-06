
import numpy as np




def batch_PID(par_batchPID,t_memory1,t_memory2,t_memory3,t_memory4,t_memory5,t_memory8,t_delta,
              So_memory1, So_memory2, So_memory3, So_memory4, So_memory5, So_memory8,
              sp_memory1, sp_memory2, sp_memory3, sp_memory4, sp_memory5, sp_memory8,
              memory_e_batch_1, memory_e_batch_2, memory_e_batch_3, memory_e_batch_4, memory_e_batch_5, memory_e_batch_8,
              u_batch_1, u_batch_2, u_batch_3, u_batch_4, u_batch_5, u_batch_8
              ):


    Kc_batchPID = 1 / 1.18  # 5#4# 0.4/240/24
    taui_batchPID = 0.25  # 5#0.25#0.3
    tauc_batchPID = 0.1  # 3#0.1


    # phase 1
    E_batch_1 = np.zeros((1, len(t_memory1)))

    tau_w1 = par_batchPID[0]
    theta_w1 = par_batchPID[1]

    # 계산 범위 구하기

    tp_1 = int(tau_w1 * 3 / t_delta)  # time period 1

    # weight function 계산
    t1_index = np.where(np.array(t_memory1) > theta_w1)[0][0]
    t1_sub2 = np.array(t_memory1[t1_index:])
    w1_sub1 = np.zeros(t1_index)
    w1_sub2 = ((t1_sub2 - theta_w1) / tau_w1) * np.exp(-((t1_sub2 - theta_w1) / tau_w1))
    w1 = np.concatenate([w1_sub1, w1_sub2])

    for t1 in range(len(t_memory1)):
        if t1 + tp_1 <= len(t_memory1):

            # E_batch(k, t) 계산
            E_batch_1[0][t1] = np.divide(sum((np.multiply(
                np.array(sp_memory1[t1:t1 + tp_1]) - np.array(So_memory1[t1:t1 + tp_1]),
                w1[t1:t1 + tp_1])) * t_delta), sum(w1[t1:t1 + tp_1] * t_delta))

        else:

            # E_batch(k, t) 계산
            E_batch_1[0][t1] = np.divide(sum((np.multiply(
                np.array(sp_memory1[t1:len(t_memory1)]) - np.array(So_memory1[t1:len(t_memory1)]),
                w1[t1:len(t_memory1)])) * t_delta), sum(w1[t1:len(t_memory1)] * t_delta))

    # phase 2
    E_batch_2 = np.zeros((1, len(t_memory2)))

    tau_w2 = par_batchPID[2]
    theta_w2 = par_batchPID[3]

    # 계산 범위 구하기

    tp_2 = int(tau_w2 * 3 / t_delta)  # time period 1

    # weight function 계산
    t2_index = np.where(np.array(t_memory2) > theta_w2)[0][0]
    t2_sub2 = np.array(t_memory2[t2_index:])
    w2_sub1 = np.zeros(t2_index)
    w2_sub2 = ((t2_sub2 - theta_w2) / tau_w1) * np.exp(-((t2_sub2 - theta_w2) / tau_w2))
    w2 = np.concatenate([w2_sub1, w2_sub2])

    for t2 in range(len(t_memory2)):
        if t2 + tp_2 <= len(t_memory2):

            # E_batch(k, t) 계산
            E_batch_2[0][t2] = np.divide(sum((np.multiply(
                np.array(sp_memory2[t2:t2 + tp_2]) - np.array(So_memory2[t2:t2 + tp_2]),
                w2[t2:t2 + tp_2])) * t_delta), sum(w2[t2:t2 + tp_2] * t_delta))

        else:

            # E_batch(k, t) 계산
            E_batch_2[0][t2] = np.divide(sum((np.multiply(
                np.array(sp_memory2[t2:len(t_memory2)]) - np.array(So_memory2[t2:len(t_memory2)]),
                w2[t2:len(t_memory2)])) * t_delta), sum(w2[t2:len(t_memory2)] * t_delta))

    # phase 3
    E_batch_3 = np.zeros((1, len(t_memory3)))

    tau_w3 = par_batchPID[4]
    theta_w3 = par_batchPID[5]

    # 계산 범위 구하기

    tp_3 = int(tau_w3 * 3 / t_delta)  # time period 1

    # weight function 계산
    t3_index = np.where(np.array(t_memory3) > theta_w3)[0][0]
    t3_sub2 = np.array(t_memory3[t3_index:])
    w3_sub1 = np.zeros(t3_index)
    w3_sub2 = ((t3_sub2 - theta_w3) / tau_w1) * np.exp(-((t3_sub2 - theta_w1) / tau_w1))
    w3 = np.concatenate([w3_sub1, w3_sub2])

    for t3 in range(len(t_memory3)):
        if t3 + tp_3 <= len(t_memory3):

            # E_batch(k, t) 계산
            E_batch_3[0][t3] = np.divide(sum((np.multiply(
                np.array(sp_memory3[t3:t3 + tp_3]) - np.array(So_memory3[t3:t3 + tp_3]),
                w3[t3:t3 + tp_3])) * t_delta), sum(w3[t3:t3 + tp_3] * t_delta))

        else:

            # E_batch(k, t) 계산
            E_batch_3[0][t3] = np.divide(sum((np.multiply(
                np.array(sp_memory3[t3:len(t_memory3)]) - np.array(So_memory3[t3:len(t_memory3)]),
                w3[t3:len(t_memory3)])) * t_delta), sum(w3[t3:len(t_memory3)] * t_delta))

    # phase 4
    E_batch_4 = np.zeros((1, len(t_memory4)))

    tau_w4 = par_batchPID[6]
    theta_w4 = par_batchPID[7]

    # 계산 범위 구하기

    tp_4 = int(tau_w4 * 3 / t_delta)  # time period 1

    # weight function 계산
    t4_index = np.where(np.array(t_memory4) > theta_w4)[0][0]
    t4_sub2 = np.array(t_memory4[t4_index:])
    w4_sub1 = np.zeros(t4_index)
    w4_sub2 = ((t4_sub2 - theta_w4) / tau_w1) * np.exp(-((t4_sub2 - theta_w4) / tau_w4))
    w4 = np.concatenate([w4_sub1, w4_sub2])

    for t4 in range(len(t_memory4)):
        if t4 + tp_4 <= len(t_memory4):

            # E_batch(k, t) 계산
            E_batch_4[0][t4] = np.divide(sum((np.multiply(
                np.array(sp_memory4[t4:t4 + tp_4]) - np.array(So_memory4[t4:t4 + tp_4]),
                w4[t4:t4 + tp_4])) * t_delta), sum(w4[t4:t4 + tp_4] * t_delta))

        else:

            # E_batch(k, t) 계산
            E_batch_4[0][t4] = np.divide(sum((np.multiply(
                np.array(sp_memory4[t4:len(t_memory4)]) - np.array(So_memory4[t4:len(t_memory4)]),
                w4[t4:len(t_memory4)])) * t_delta), sum(w4[t4:len(t_memory4)] * t_delta))

    # phase 5
    E_batch_5 = np.zeros((1, len(t_memory5)))

    tau_w5 = par_batchPID[8]
    theta_w5 = par_batchPID[9]

    # 계산 범위 구하기

    tp_5 = int(tau_w5 * 3 / t_delta)  # time period 1

    # weight function 계산
    t5_index = np.where(np.array(t_memory5) > theta_w5)[0][0]
    t5_sub2 = np.array(t_memory5[t5_index:])
    w5_sub1 = np.zeros(t5_index)
    w5_sub2 = ((t5_sub2 - theta_w5) / tau_w5) * np.exp(-((t5_sub2 - theta_w5) / tau_w5))
    w5 = np.concatenate([w5_sub1, w5_sub2])

    for t5 in range(len(t_memory5)):
        if t5 + tp_5 <= len(t_memory5):

            # E_batch(k, t) 계산
            E_batch_5[0][t5] = np.divide(sum((np.multiply(
                np.array(sp_memory5[t5:t5 + tp_5]) - np.array(So_memory5[t5:t5 + tp_5]),
                w5[t5:t5 + tp_5])) * t_delta), sum(w5[t5:t5 + tp_5] * t_delta))

        else:

            # E_batch(k, t) 계산
            E_batch_5[0][t5] = np.divide(sum((np.multiply(
                np.array(sp_memory5[t5:len(t_memory5)]) - np.array(So_memory5[t5:len(t_memory5)]),
                w5[t5:len(t_memory5)])) * t_delta), sum(w5[t5:len(t_memory5)] * t_delta))

    # phase 8
    E_batch_8 = np.zeros((1, len(t_memory8)))

    tau_w8 = par_batchPID[14]
    theta_w8 = par_batchPID[15]

    # 계산 범위 구하기

    tp_8 = int(tau_w8 * 3 / t_delta)  # time period 1

    # weight function 계산
    t8_index = np.where(np.array(t_memory8) > theta_w8)[0][0]
    t8_sub2 = np.array(t_memory8[t8_index:])
    w8_sub1 = np.zeros(t8_index)
    w8_sub2 = ((t8_sub2 - theta_w8) / tau_w8) * np.exp(-((t8_sub2 - theta_w8) / tau_w8))
    w8 = np.concatenate([w8_sub1, w8_sub2])

    for t8 in range(len(t_memory8)):
        if t8 + tp_8 <= len(t_memory8):

            # E_batch(k, t) 계산
            E_batch_8[0][t8] = np.divide(sum((np.multiply(
                np.array(sp_memory8[t8:t8 + tp_8]) - np.array(So_memory8[t8:t8 + tp_8]),
                w8[t8:t8 + tp_8])) * t_delta), sum(w8[t8:t8 + tp_8] * t_delta))

        else:

            # E_batch(k, t) 계산
            E_batch_8[0][t8] = np.divide(sum((np.multiply(
                np.array(sp_memory8[t8:len(t_memory8)]) - np.array(So_memory8[t8:len(t_memory8)]),
                w8[t8:len(t_memory8)])) * t_delta), sum(w8[t8:len(t_memory8)] * t_delta))

    memory_e_batch_1 = np.append(memory_e_batch_1, E_batch_1, axis=0)
    memory_e_batch_2 = np.append(memory_e_batch_2, E_batch_2, axis=0)
    memory_e_batch_3 = np.append(memory_e_batch_3, E_batch_3, axis=0)
    memory_e_batch_4 = np.append(memory_e_batch_4, E_batch_4, axis=0)
    memory_e_batch_5 = np.append(memory_e_batch_5, E_batch_5, axis=0)
    memory_e_batch_8 = np.append(memory_e_batch_8, E_batch_8, axis=0)

    ie_batchPID_1 = memory_e_batch_1[:, :].sum(axis=0)
    de_batchPID_1 = memory_e_batch_1[-1, :] - memory_e_batch_1[-2, :]

    ie_batchPID_2 = memory_e_batch_2[:, :].sum(axis=0)
    de_batchPID_2 = memory_e_batch_2[-1, :] - memory_e_batch_2[-2, :]

    ie_batchPID_3 = memory_e_batch_3[:, :].sum(axis=0)
    de_batchPID_3 = memory_e_batch_3[-1, :] - memory_e_batch_3[-2, :]

    ie_batchPID_4 = memory_e_batch_4[:, :].sum(axis=0)
    de_batchPID_4 = memory_e_batch_4[-1, :] - memory_e_batch_4[-2, :]

    ie_batchPID_5 = memory_e_batch_5[:, :].sum(axis=0)
    de_batchPID_5 = memory_e_batch_5[-1, :] - memory_e_batch_5[-2, :]

    ie_batchPID_8 = memory_e_batch_8[:, :].sum(axis=0)
    de_batchPID_8 = memory_e_batch_8[-1, :] - memory_e_batch_8[-2, :]

    P_batchPID_1 = Kc_batchPID * memory_e_batch_1[-1, :]
    I_batchPID_1 = Kc_batchPID * (1 / taui_batchPID) * ie_batchPID_1
    D_batchPID_1 = Kc_batchPID * tauc_batchPID * de_batchPID_1

    P_batchPID_2 = Kc_batchPID * memory_e_batch_2[-1, :]
    I_batchPID_2 = Kc_batchPID * (1 / taui_batchPID) * ie_batchPID_2
    D_batchPID_2 = Kc_batchPID * tauc_batchPID * de_batchPID_2

    P_batchPID_3 = Kc_batchPID * memory_e_batch_3[-1, :]
    I_batchPID_3 = Kc_batchPID * (1 / taui_batchPID) * ie_batchPID_3
    D_batchPID_3 = Kc_batchPID * tauc_batchPID * de_batchPID_3

    P_batchPID_4 = Kc_batchPID * memory_e_batch_4[-1, :]
    I_batchPID_4 = Kc_batchPID * (1 / taui_batchPID) * ie_batchPID_4
    D_batchPID_4 = Kc_batchPID * tauc_batchPID * de_batchPID_4

    P_batchPID_5 = Kc_batchPID * memory_e_batch_5[-1, :]
    I_batchPID_5 = Kc_batchPID * (1 / taui_batchPID) * ie_batchPID_5
    D_batchPID_5 = Kc_batchPID * tauc_batchPID * de_batchPID_5

    P_batchPID_8 = Kc_batchPID * memory_e_batch_8[-1, :]
    I_batchPID_8 = Kc_batchPID * (1 / taui_batchPID) * ie_batchPID_8
    D_batchPID_8 = Kc_batchPID * tauc_batchPID * de_batchPID_8

    u_batchPID_1 = np.asarray([P_batchPID_1 + I_batchPID_1 + D_batchPID_1])
    u_batchPID_2 = np.asarray([P_batchPID_2 + I_batchPID_2 + D_batchPID_2])
    u_batchPID_3 = np.asarray([P_batchPID_3 + I_batchPID_3 + D_batchPID_3])
    u_batchPID_4 = np.asarray([P_batchPID_4 + I_batchPID_4 + D_batchPID_4])
    u_batchPID_5 = np.asarray([P_batchPID_5 + I_batchPID_5 + D_batchPID_5])
    u_batchPID_8 = np.asarray([P_batchPID_8 + I_batchPID_8 + D_batchPID_8])

    u_batch_1 = np.append(u_batch_1, u_batchPID_1, axis=0)
    u_batch_2 = np.append(u_batch_2, u_batchPID_2, axis=0)
    u_batch_3 = np.append(u_batch_3, u_batchPID_3, axis=0)
    u_batch_4 = np.append(u_batch_4, u_batchPID_4, axis=0)
    u_batch_5 = np.append(u_batch_5, u_batchPID_5, axis=0)
    u_batch_8 = np.append(u_batch_8, u_batchPID_8, axis=0)

    return u_batch_1, u_batch_2, u_batch_3, u_batch_4, u_batch_5, u_batch_8, memory_e_batch_1, memory_e_batch_2, memory_e_batch_3, memory_e_batch_4, memory_e_batch_5, memory_e_batch_8

