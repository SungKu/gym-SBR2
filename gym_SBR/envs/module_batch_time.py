import numpy as np

def batch_time(t_cycle, t_ratio, t_delta):

    t_phs1 = (t_cycle) * t_ratio[0]  # around 30 min
    t_phs2 = (t_cycle) * t_ratio[1]
    t_phs3 = (t_cycle) * t_ratio[2]
    t_phs4 = (t_cycle) * t_ratio[3]
    t_phs5 = (t_cycle) * t_ratio[4]
    t_phs6 = (t_cycle) * t_ratio[5]
    t_phs7 = (t_cycle) * t_ratio[6]
    t_phs8 = (t_cycle) * t_ratio[7]

    t_start = 0
    t_end = 0

    t_memory1 =  []
    t_memory2 =  []
    t_memory3 =  []
    t_memory4 =  []
    t_memory5 =  []
    t_memory6 =  []
    t_memory7 =  []
    t_memory8 =  []

    t_start = t_end
    t_end = t_start + t_phs1
    t_save1 = np.linspace(t_start, t_end,int( (t_end - t_start) / (t_delta * 10)))
    t_memory1.append(t_save1[0])
    for i in range(len(t_save1) - 1):
          t_s_flag = t_save1[i]
          t_e_flag = t_save1[i + 1]
          t_range = np.linspace(t_s_flag, t_e_flag, int((t_e_flag - t_s_flag) / t_delta))
          for ii in range(len(t_range) - 1):
                t_memory1.append(t_range[ii + 1])


    t_start = t_end + t_delta
    t_end = t_start + t_phs2
    t_save2 = np.linspace(t_start, t_end, int((t_end - t_start) / (t_delta * 10)))
    t_memory2.append(t_save2[0])
    for i in range(len(t_save2) - 1):
          t_s_flag = t_save2[i]
          t_e_flag = t_save2[i + 1]
          t_range = np.linspace(t_s_flag, t_e_flag, int((t_e_flag - t_s_flag) / t_delta))
          for ii in range(len(t_range) - 1):
                t_memory2.append(t_range[ii + 1])

    t_start = t_end + t_delta
    t_end = t_start + t_phs3
    t_save3 = np.linspace(t_start, t_end, int((t_end - t_start) / (t_delta * 10)))
    t_memory3.append(t_save3[0])
    for i in range(len(t_save3) - 1):
          t_s_flag = t_save3[i]
          t_e_flag = t_save3[i + 1]
          t_range = np.linspace(t_s_flag, t_e_flag,int((t_e_flag - t_s_flag) / t_delta))
          for ii in range(len(t_range) - 1):
                t_memory3.append(t_range[ii + 1])

    t_start = t_end + t_delta
    t_end = t_start + t_phs4
    t_save4 = np.linspace(t_start, t_end, int((t_end - t_start) / (t_delta * 10)))
    t_memory4.append(t_save4[0])
    for i in range(len(t_save4) - 1):
          t_s_flag = t_save4[i]
          t_e_flag = t_save4[i + 1]
          t_range = np.linspace(t_s_flag, t_e_flag, int(((t_e_flag - t_s_flag) / t_delta)))
          for ii in range(len(t_range) - 1):
                t_memory4.append(t_range[ii + 1])

    t_start = t_end + t_delta
    t_end = t_start + t_phs5
    t_save5 = np.linspace(t_start, t_end, int((t_end - t_start) / (t_delta * 10)))
    t_memory5.append(t_save5[0])
    for i in range(len(t_save5) - 1):
          t_s_flag = t_save5[i]
          t_e_flag = t_save5[i + 1]
          t_range = np.linspace(t_s_flag, t_e_flag, int((t_e_flag - t_s_flag) / t_delta))
          for ii in range(len(t_range) - 1):
                t_memory5.append(t_range[ii + 1])

    t_start = t_end + t_delta
    t_end = t_start + t_phs6
    t_save6 = np.linspace(t_start, t_end, int((t_end - t_start) / (t_delta * 10)))
    t_memory6.append(t_save6[0])
    for i in range(len(t_save6) - 1):
          t_s_flag = t_save6[i]
          t_e_flag = t_save6[i + 1]
          t_range = np.linspace(t_s_flag, t_e_flag, int((t_e_flag - t_s_flag) / t_delta))
          for ii in range(len(t_range) - 1):
                t_memory6.append(t_range[ii + 1])

    t_start = t_end + t_delta
    t_end = t_start + t_phs7
    t_save7 = np.linspace(t_start, t_end, int((t_end - t_start) / (t_delta * 10)))
    t_memory7.append(t_save7[0])
    for i in range(len(t_save7) - 1):
          t_s_flag = t_save7[i]
          t_e_flag = t_save7[i + 1]
          t_range = np.linspace(t_s_flag, t_e_flag, int((t_e_flag - t_s_flag) / t_delta))
          for ii in range(len(t_range) - 1):
                t_memory7.append(t_range[ii + 1])

    t_start = t_end + t_delta
    t_end = t_start + t_phs8
    t_save8 = np.linspace(t_start, t_end, int((t_end - t_start) / (t_delta * 10)))
    t_memory8.append(t_save8[0])
    for i in range(len(t_save8) - 1):
          t_s_flag = t_save8[i]
          t_e_flag = t_save8[i + 1]
          t_range = np.linspace(t_s_flag, t_e_flag, int((t_e_flag - t_s_flag) / t_delta))
          for ii in range(len(t_range) - 1):
                t_memory8.append(t_range[ii + 1])


    return t_memory1, t_memory2, t_memory3, t_memory4, t_memory5, t_memory6, t_memory7, t_memory8

