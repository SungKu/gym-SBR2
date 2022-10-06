import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec

def components(t, x,epoch_steps):
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(12)

    x = x.T
    x1 = x[0,:]
    x2 =x[1,:]
    x3 =x[2,:]
    x4 =x[3,:]
    x5 =x[4,:]
    x6 =x[5,:]
    x7 =x[6,:]
    x8 =x[7,:]
    x9 =x[8,:]
    x10 =x[9,:]
    x11 =x[10,:]
    x12 =x[11,:]
    x13 =x[12,:]
    x14 =x[13,:]




    # Working Volume
    plt.subplot(3,5,1)
    plt.plot(t, x1, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('L')
    plt.title('Volume')


    # Si
    plt.subplot(3,5,2)
    plt.plot(t, x2, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Si')
    # Ss
    plt.subplot(3,5,3)
    plt.plot(t, x3, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Ss')
    # Si
    plt.subplot(3,5,4)
    plt.plot(t, x4, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Xi')
    # Xs
    plt.subplot(3,5,5)
    plt.plot(t, x5, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Xs')
    # Xbh
    plt.subplot(3,5,6)
    plt.plot(t, x6, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Xbh')
    # Xba
    plt.subplot(3,5,7)
    plt.plot(t, x7, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Xba')
    # Xp
    plt.subplot(3,5,8)
    plt.plot(t, x8, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Xp')


    # So
    plt.subplot(3,5,9)
    plt.ylim(0, 7.5)
    plt.plot(t, x9, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('So')


    # Sno
    plt.subplot(3,5,10)
    plt.plot(t, x10, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Sno')

    # Snh
    plt.subplot(3,5,11)
    plt.plot(t, x11, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Snh')
    # Snd
    plt.subplot(3,5,12)
    plt.plot(t, x12, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Snd')
    # Xnd
    plt.subplot(3,5,13)
    plt.plot(t, x13, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Xnd')
    # Salk
    plt.subplot(3,5,14)
    plt.plot(t, x14, 'b-')
    plt.xlabel('Time (day)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Salk')

    plt.savefig('savefig_default'+str(epoch_steps)+'.png')
    #plt.show()
    plt.close()


def reward_history(reward_history):
    plt.plot(reward_history, 'k-')
    plt.xlabel('episodes')
    plt.ylabel('rewards')

    plt.close()
