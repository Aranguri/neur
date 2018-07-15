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
        matlab = scipy.io.loadmat('{}.mat'.format(file))
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

    for i in range(0, 5):
        vm_vector = np.mean(data[i]['Vm'].reshape(-1, bin_size), axis=1)
        fl_vector = np.mean(data[i]['Fluorescence'].reshape(-1, bin_size), axis=1)
        vm_vector = (vm_vector - np.mean(vm_vector)) / (np.std(vm_vector) * len(vm_vector))
        fl_vector = (fl_vector - np.mean(fl_vector)) / (np.std(fl_vector))

        correlation = np.correlate(vm_vector, fl_vector, 'same')
        half = len(correlation) / 2
        correlation = correlation[int(half - window):int(half + window)]
        time = np.arange(-len(correlation) / 2, len(correlation) / 2, 1) * planck_time * bin_size
    plt.plot(time, correlation)
    plt.show()

def correlate2():
    bin_size = 100
    outer_window = 10 / (planck_time * bin_size)
    inner_window = outer_window / 100
    sum = 0
    real_corrs, fake_corrs = [[], []]

    for i in range(0, 5):
        vm_vector = np.mean(data[i]['Vm'].reshape(-1, bin_size), axis=1)
        vm_vector = (vm_vector - np.mean(vm_vector)) / (np.std(vm_vector) * len(vm_vector))
        for j in range(0, 5):
            fl_vector = np.mean(data[j]['Fluorescence'].reshape(-1, bin_size), axis=1)
            fl_vector = (fl_vector - np.mean(fl_vector)) / (np.std(fl_vector))
            correlation = np.correlate(vm_vector, fl_vector, 'same')
            half = len(correlation) / 2
            inner = correlation[int(half - inner_window):int(half + inner_window)]
            first = correlation[int(half - outer_window):int(half - inner_window)]
            second = correlation[int(half + inner_window):int(half + outer_window)]
            inner = np.average(inner)
            outer = np.average(np.concatenate((first, second)))

            if i == j: real_corrs.append(inner - outer)
            else: fake_corrs.append(inner - outer)

    print(np.std(real_corrs))
    print(np.std(fake_corrs))

    plt.scatter(np.zeros(len(fake_corrs)), fake_corrs)
    plt.scatter([0], [np.average(fake_corrs)])
    plt.scatter(np.ones(len(real_corrs)), real_corrs)
    plt.scatter([1], [np.average(real_corrs)])
    plt.show()
    #print (real_corrs)
    #print (fake_corrs)


def plot_data(smoothed=False):
    window = 20
    vectors = data[0]
    vm_vector = vectors['Vm']#[0:20000]
    fl_vector = vectors['Fluorescence']#[0:20000]
    time_vector = vectors['Time']#[0:(20000 - window)]

    if smoothed:
        vm_vector = [np.average(vm_vector[(i - window):i]) for i in range(window, len(vm_vector))]
        fl_vector = [np.average(fl_vector[(i - window):i]) for i in range(window, len(fl_vector))]

    plot_two_figs(vm_vector, fl_vector, time_vector)
    plt.show()

def wt_test():
    vectors = data[1]
    vm = vectors['Vm'][0:10000]
    fl = vectors['Fluorescence'][0:10000]
    time = vectors['Time'][0:10000]
    plot_two_figs(vm, fl, time)

    coefs, freqs = pywt.cwt(vm, np.arange(1, 129), 'mexh', .0025)
    freqs = [int(freq * 100) for freq in freqs]

    j = 0
    new_coefs = [[] for _ in range(10000)]
    for i in range(9999, -1, -1):
        new_coefs[i] = coefs[j]
        if i in freqs and j != 128:
            j += 1


    plt.matshow(new_coefs, aspect='auto')
    plt.show()

def plot_two_figs(vector1, vector2, time):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time, vector1)
    plt.subplot(212)
    plt.plot(time, vector2)

init()
correlate2()
#plot_data()
#covariance()
#wt_test()
