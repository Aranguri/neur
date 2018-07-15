def pywavelet():
    vector = data[0]['Vm'][0:1024]
    widths = np.arange(1, 31)
    cwtmatr, freqs = pywt.cwt(vector, widths, 'mexh')
    plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
        vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.show()

def sciwavelet():
    sig = data[0]['Vm'][0:1024]
    widths = np.arange(1, 31)
    cwtmatr = signal.cwt(sig, signal.ricker, widths)
    plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
        vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.show()

def correlate():
    size = 20000
    for j in range(0, 5):
        vectors = data[j]
        fl_vector = vectors['Fluorescence'][0:size]
        vm_vector = vectors['Vm'][0:size]
        print (np.correlate(fl_vector, vm_vector))
