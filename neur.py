import time as time_mod
import pycorrelate as pyc
import scipy.io
import pywt
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from scipy import signal

#cut borders, use time vector

data = []
files = ['2018-04-10-001, GCaMP6s', '2018-04-17-002, GCaMP6s', '2018-05-21-001, GCaMP6f', '2018-05-21-002, GCaMP6f', '2018-06-22-001, GCaMP7f']
planck_time = 0.0001

def init():
    for i, file in enumerate(files):
        matlab = scipy.io.loadmat('data/{}.mat'.format(file))
        vectors = matlab['Recording'][0][0]
        names = ['Vm', 'WhiskerStimulation', 'Fluorescence', 'Pupilsize', 'Walking', 'Whisking', 'Stepping', 'Time']
        data.append({})

        for name, vector in zip(names, vectors):
            if name != 'Pupilsize':
                data[i][name] = vector[0]

def covariance():
    window = 10000
    definition = 100

    for j in range(0, 5):
        vectors = data[j]
        size = len(vectors['Vm']) - window
        fl_vector = vectors['Fluorescence'][0:size + window]
        vm_vector = vectors['Vm'][0:size + window]
        fl_vector = fl_vector - np.average(fl_vector)
        vm_vector = vm_vector - np.average(vm_vector)
        covs = []
        for i in np.arange(-window, window, definition):
            if i > 0:
                vm_vector = vectors['Vm'][0:size]
                fl_vector = vectors['Fluorescence'][i:(size + i)]
            else:
                vm_vector = vectors['Vm'][-i:(size - i)]
                fl_vector = vectors['Fluorescence'][0:size]
            #cov = np.corrcoef(vm_vector, fl_vector)
            cov = np.correlate(vm_vector, fl_vector, mode='full')
            covs.append(cov[0][1])
        plt.plot(np.arange(-window, window, definition) * planck_time, covs)
    plt.show()

def correlate1():
    bin_size = 100
    window = 10 / (planck_time * bin_size)
    sum = 0

    for i in range(3, 4):
    #for i, j in [(0, 2), (1, 0), (2, 1), (3, 4), (4, 3)]:
        correlation = correlate(data[i]['Vm'], data[i]['Fluorescence'], bin_size)
        half = len(correlation) / 2
        correlation = correlation[int(half - window):int(half + window)]
        time = np.arange(-len(correlation) / 2, len(correlation) / 2, 1) * planck_time * bin_size
        plt.plot(time, correlation)
    plt.legend(files)
    plt.xlabel('Delay from axon activity to cell activity (s)')
    plt.ylabel('Correlation')
    plt.show()

def correlate2():
    bin_size = 100
    outer_window = 10 / (planck_time * bin_size)
    inner_window = outer_window / 50
    sum = 0
    real_corrs, fake_corrs = [[], []]

    for i in range(0, 5):
        for j in range(0, 5):
            correlation = correlate(data[i]['Vm'], data[j]['Fluorescence'], bin_size)
            half = len(correlation) / 2
            inner = correlation[int(half - inner_window):int(half + inner_window)]
            first = correlation[int(half - outer_window):int(half - inner_window)]
            second = correlation[int(half + inner_window):int(half + outer_window)]
            inner = np.average(inner)
            outer = np.average(np.concatenate((first, second)))

            if i == j: real_corrs.append(inner - outer)
            else: fake_corrs.append(inner - outer)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.scatter(np.random.uniform(.95, 1.05, len(fake_corrs)), fake_corrs, 12)
    ax1.boxplot(fake_corrs, showfliers=False)
    ax1.set_xticklabels([])
    ax1.set_xlabel('Shuffled data')
    ax1.set_ylabel('Correlation bump')
    ax2.scatter(np.random.uniform(.95, 1.05, len(real_corrs)), real_corrs, 12, 'r')
    ax2.boxplot(real_corrs, showfliers=False)
    ax2.set_xticklabels([])
    ax2.set_xlabel('Real data')
    plt.show()

def plot_data(smoothed=False):
    window = 100
    vectors = data[2]
    vm_vector = vectors['Vm']
    fl_vector = vectors['Fluorescence']
    time_vector = vectors['Time'][0:-window]

    if smoothed:
        vm_vector = [np.average(vm_vector[(i - window):i]) for i in range(window, len(vm_vector))]
        fl_vector = [np.average(fl_vector[(i - window):i]) for i in range(window, len(fl_vector))]

    plot_two_figs(vm_vector, fl_vector, time_vector)
    plt.show()

def wt_range(start, end):
    #return np.arange(2500 / end, 2500 / start, 100)
    #return np.logspace(1.4, 3.7)
    return np.logspace(2.4, 4.4)

def wt_test():
    vectors = data[0]
    size = len(vectors['Vm'])
    padding = int(size / 10)
    vm = vectors['Vm'][0:size]
    fl = vectors['Fluorescence'][0:size]
    time = vectors['Time'][0:size]
    #plot_two_figs(vm, fl, time)
    print(size)

    coefs, freqs = pywt.cwt(vm, wt_range(.5, 100), 'mexh', .0001)
    print (freqs)
    flat_coefs = np.sum(coefs, axis=0)
    plt.plot(flat_coefs)
    plt.show()
    '''
    print (freqs)

    freqs = [f * 10 for f in np.flip(freqs, 0)]
    new_coefs = []#[[] for _ in range(10000)]
    j = -1
    current_freq = -1
    for i in range(4, 1000):
        if i > current_freq and j < (len(coefs) - 1):
            j += 1
            current_freq = freqs[j]
        new_coefs.append(coefs[j])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(new_coefs, interpolation='nearest', aspect='auto')

    #flat_coefs100 = np.sum(coefs100, axis=0)
    #flat_coefs10 = np.sum(coefs10, axis=0)

    #plt.matshow(coefs10)
    #plt.plot(flat_coefs30[padding:-padding])
    #plt.plot(flat_coefs10[padding:-padding])
    #plt.plot(flat_coefs100[padding:-padding])
    #plt.legend(['0-30 Hz', '30-100 Hz'])
    plt.show(block=False)
    time_mod.sleep(10000)

    plt.figure(1)
    plt.matshow(coefs1)
    plt.yticks([0])#30, 31, 33, 36, 42, 50, 65, 78, 100])
    plt.figure(2)
    plt.matshow(coefs2)
    plt.yticks([0])#0, 1, 2, 4, 8, 16, 30])
    plt.show()
    #ax.set_xticklabels(np.arange(0, size*planck_time, planck_time * 1000))
    #ax.set_yticklabels(freqs2)

    #print (np.shape(coefs), np.shape(freqs))
    freqs = [int(freq * 100) for freq in freqs]

    j = 0
    new_coefs = [[] for _ in range(10000)]
    for i in range(9999, -1, -1):
        new_coefs[i] = coefs[j]
        if i in freqs and j != 128:
            j += 1

    plt.matshow(new_coefs, aspect='auto')
    plt.show()
    '''

def correlate(array1, array2, bin_size):
    array1 = np.mean(array1.reshape(-1, bin_size), axis=1)
    array2 = np.mean(array2.reshape(-1, bin_size), axis=1)
    array1 = (array1 - np.mean(array1)) / (np.std(array1) * len(array1))
    array2 = (array2 - np.mean(array2)) / (np.std(array2))
    print (len(array1))
    print (len(array2))
    return np.correlate(array1, array2, 'same')

def plot_two_figs(vector1, vector2, time):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time, vector1)
    plt.subplot(212)
    plt.plot(time, vector2)

def near_zero():
    window = .5 / planck_time
    bin_size = 10
    window = int(window / bin_size)
    for i in range(4, 5):
        correlation = correlate(data[i]['Vm'], data[i]['Fluorescence'], bin_size)
        half = int(len(correlation) / 2)
        correlation = correlation[half - window:half + window]
        best_point = -100
        best_time = None
        for j, point in enumerate(correlation):
            if point > best_point:
                best_point = point
                best_time = j
        print (i, best_time - window, best_point)
        plt.plot(correlation)
    plt.show()

init()
#correlate1()
#plot_data(True)
#covariance()
wt_test()
#near_zero()


#Make an average of the columns
