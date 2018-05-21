import argparse
import copy
import json
import pickle
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np


class SignalType(str, Enum):
    SIN = 'sin'
    SIN_SINGLE = 'sin_single'
    SIN_DOUBLE = 'sin_double'
    RECT = 'rect'
    RECT_SIMETRIC = 'rect_sim'
    TRI = 'tri'
    STEP = 'step'
    UNIT_IMPULSE = 'unit_impulse'
    NOISE_NORM = 'noise_norm'
    NOISE_GAUSS = 'noise_gauss'
    NOISE_IMPULSE = 'noise_impulse'


class SignalParams:
    def __init__(self, amplitude=0.0, time_start=0.0, duration=0.0,
                 peroid=0.0, duty=0.0, sampfq=0.0, sigtype=None, offset=False):
        self.amplitude = amplitude
        self.time_start = time_start
        self.duration = duration
        self.period = peroid
        self.duty = duty
        self.sampfq = sampfq
        self.type = sigtype
        self.offset = offset


class Signal:
    def __init__(self, x=None, y=None, params=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.params_hist = []
        if isinstance(params, list):
            self.params_hist = params
        else:
            self.params_hist.append(('root', params))


def to_json(pyobj, filename):
    with open(filename, 'w') as f:
        json.dump(pyobj.__dict__, f, indent=2)


def to_bin(pyobj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pyobj, f)


def sig_to_txt(sig, filename):
    with open(filename, 'w') as f:
        for i in range(len(sig.x)):
            f.write("{0}\t\t{1}\n".format(sig.x[i], sig.y[i]))


def from_json(filename):
    with open(filename, 'r') as f:
        js = f.read()
        return json.loads(js)


def from_bin(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def gen_x(params):
    return np.arange(params.time_start, params.duration + params.time_start, 1 / params.sampfq)


def sig_sin(params):
    x = gen_x(params)
    if params.offset:
        x = np.arange(0, params.duration, 1 / params.sampfq)
        y = params.amplitude * np.sin(2 * np.pi / params.period * (x + params.time_start))
    else:
        y = params.amplitude * np.sin(2 * np.pi / params.period * (x - params.time_start))
    return Signal(x, y, params)


def sig_sin_single(params):
    x = gen_x(params)
    y = params.amplitude * np.sin(2 * np.pi / params.period * (x - params.time_start))
    y[y < 0] = 0
    return Signal(x, y, params)


def sig_sin_double(params):
    x = gen_x(params)
    y = params.amplitude * np.sin(2 * np.pi / params.period * (x - params.time_start))
    y = abs(y)
    return Signal(x, y, params)


def sig_tri(params):
    x = gen_x(params)
    y = []
    cnt = 0
    for i in range(len(x)):
        if x[i] > params.period * cnt + params.period + params.time_start:
            cnt += 1
        if x[i] < params.period * params.duty + cnt * params.period + params.time_start:
            y.append(
                (params.amplitude * (x[i] - cnt * params.period - params.time_start))
                / (params.duty * params.period))
        else:
            y.append(
                (-params.amplitude * (x[i] - cnt * params.period - params.time_start))
                / (params.period * (1 - params.duty))
                + params.amplitude / (1 - params.duty))
    return Signal(x, y, params)


def sig_rect(params):
    x = gen_x(params)
    y = []
    cnt = 0
    for i in range(len(x)):
        if x[i] > params.period * cnt + params.period + params.time_start:
            cnt += 1
        if x[i] < params.period * params.duty + cnt * params.period + params.time_start:
            y.append(params.amplitude)
        else:
            y.append(0) if params.type == SignalType.RECT else y.append(-params.amplitude)
    return Signal(x, y, params)


def sig_step(params):
    x = gen_x(params)
    y = []
    for i in range(len(x)):
        if x[i] < params.period + params.time_start:
            y.append(0)
        elif x[i] == params.period + params.time_start:
            y.append(params.amplitude / 2)
        else:
            y.append(params.amplitude)
    return Signal(x, y, params)


def sig_impulse(params):
    eps = 10E-5
    x = gen_x(params)
    y = []
    for i in range(len(x)):
        if abs(x[i] - params.period - params.time_start) < eps:
            y.append(params.amplitude)
        else:
            y.append(0)
    return Signal(x, y, params)


def noise_normal(params):
    x = gen_x(params)
    y = np.random.uniform(-params.amplitude, params.amplitude, len(x))
    return Signal(x, y, params)


def noise_gauss(params):
    x = gen_x(params)
    y = np.random.normal(0, params.amplitude, len(x))
    return Signal(x, y, params)


def noise_impulse(params):
    x = gen_x(params)
    y = []
    for i in range(len(x)):
        if np.random.random() < params.duty:
            y.append(params.amplitude)
        else:
            y.append(0)
    return Signal(x, y, params)


def sig_add(sig1, sig2):
    hist = sig1.params_hist
    hist.append(('+', sig2.params_hist))
    return Signal(sig1.x, np.add(sig1.y, sig2.y), hist)


def sig_sub(sig1, sig2):
    hist = sig1.params_hist
    hist.append(('-', sig2.params_hist))
    return Signal(sig1.x, np.subtract(sig1.y, sig2.y), hist)


def sig_mul(sig1, sig2):
    hist = sig1.params_hist
    hist.append(('*', sig2.params_hist))
    return Signal(sig1.x, np.multiply(sig1.y, sig2.y), hist)


def sig_div(sig1, sig2):
    hist = sig1.params_hist
    hist.append(('/', sig2.params_hist))
    eps = 10E-6
    y = []
    for i in range(len(sig1.y)):
        if abs(sig2.y[i]) < eps:
            y.append(np.inf)
        else:
            y.append(sig1.y[i] / sig2.y[i])
    return Signal(sig1.x, y, hist)


def avg(sig):
    return np.sum(sig.y) / len(sig.y)


def avg_abs(sig):
    y = abs(sig.y)
    return np.sum(y / len(sig.y))


def rms(sig):
    return np.sqrt(avg_pow(sig))


def variance(sig):
    x = avg(sig)
    return sum((y - x) ** 2 for y in sig.y) / len(sig.y)


def avg_pow(sig):
    return sum(y * y for y in sig.y) / len(sig.y)


def mse(sig, sig_origin):
    s = 0
    for i in range(len(sig.y)):
        s += (sig_origin.y[i] - sig.y[i]) ** 2
    return s / len(sig.y)


def snr(sig, sig_origin):
    s1 = sum(y * y for y in sig_origin.y)
    s2 = 0
    for i in range(len(sig.y)):
        s2 += (sig_origin.y[i] - sig.y[i]) ** 2
    return 10 * np.log10(s1 / s2)


def psnr(sig, sig_origin):
    return 10 * np.log10(max(sig_origin.y) / mse(sig, sig_origin))


def md(sig, sig_origin):
    dif = 0
    for i in range(len(sig.x)):
        if dif < abs(sig.y[0] - sig_origin.y[0]):
            dif = abs(sig.y[0] - sig_origin.y[0])
    return dif


def print_stats(sig):
    sig = trunc(sig)
    noinf = np.inf not in sig.y
    print('Average value: ', avg(sig) if noinf else '+-inf')
    print('Average absolute value: ', avg_abs(sig) if noinf else '+-inf')
    print('Root mean square: ', rms(sig) if noinf else '+-inf')
    print('Variance:', variance(sig) if noinf else '+-inf')
    print('Average power: ', avg_pow(sig) if noinf else '+-inf')


def print_stats_quant(sig, sig_origin):
    print('MSE: ', mse(sig, sig_origin))
    print('SNR: ', snr(sig, sig_origin), ' dB')
    print('PSNR: ', psnr(sig, sig_origin), ' dB')
    print('MD: ', md(sig, sig_origin))


def read(path):
    words = path.split('.')
    ext = words.pop()
    if ext == 'json':
        params = from_json(path)
        return gen_signal(params)
    else:
        return from_bin(path)


def gen_signal(p):
    params = SignalParams(p['amplitude'], p['time_start'], p['duration'],
                          p['period'], p['duty'], p['sampfq'], p['type'])
    return gen_signal_from_params(params)


def gen_signal_from_params(params):
    if params.type == SignalType.SIN:
        return sig_sin(params)
    if params.type == SignalType.SIN_SINGLE:
        return sig_sin_single(params)
    if params.type == SignalType.SIN_DOUBLE:
        return sig_sin_double(params)
    if params.type == SignalType.RECT:
        return sig_rect(params)
    if params.type == SignalType.RECT_SIMETRIC:
        return sig_rect(params)
    if params.type == SignalType.TRI:
        return sig_tri(params)
    if params.type == SignalType.STEP:
        return sig_step(params)
    if params.type == SignalType.UNIT_IMPULSE:
        return sig_impulse(params)
    if params.type == SignalType.NOISE_NORM:
        return noise_normal(params)
    if params.type == SignalType.NOISE_GAUSS:
        return noise_gauss(params)
    if params.type == SignalType.NOISE_IMPULSE:
        return noise_impulse(params)


def sig_op(sig1, sig2, op):
    if op == '+':
        return sig_add(sig1, sig2)
    if op == '-':
        return sig_sub(sig1, sig2)
    if op == 'x':
        return sig_mul(sig1, sig2)
    if op == '/':
        return sig_div(sig1, sig2)


def trunc(sig):
    if len(sig.params_hist) == 1:
        p = sig.params_hist[0][1]
        if p.type == SignalType.SIN or p.type == SignalType.SIN_SINGLE \
                or p.type == SignalType.SIN_DOUBLE or p.type == SignalType.RECT \
                or p.type == SignalType.RECT_SIMETRIC or p.type == SignalType.TRI:
            if p.duration > p.period:
                x = np.arange(p.time_start, p.period + p.time_start, 1 / p.sampfq)
                y = sig.y[:len(x)]
                return Signal(x, y, sig.params_hist)
    return sig


def sample(sig, fq):
    x = []
    y = []
    prev_fq = len(sig.x) / (sig.x[-1] - sig.x[0])
    step = int(prev_fq // fq)
    for i in range(0, len(sig.x), step):
        x.append(sig.x[i])
        y.append(sig.y[i])
    return Signal(x, y, sig.params_hist)


def quant(sig, new_min, new_max, round_flag):
    old_min = min(sig.y)
    old_max = max(sig.y)
    old_range = np.abs(old_max - old_min)
    new_range = np.abs(new_max - new_min)
    if round_flag:
        y = np.round(((sig.y - old_min) * new_range / old_range) + new_min)
    else:
        y = np.round(((sig.y - old_min) * new_range / old_range) + new_min)
    return Signal(sig.x, y, sig.params_hist)


def dequant_sinc(sig, new_fq):
    x = np.arange(sig.x[0], sig.x[-1], 1 / new_fq)
    y = []
    peroid = sig.x[1] - sig.x[0]
    for i in range(len(x)):
        y.append(np.sum(sig.y * np.sinc((x[i] - sig.x) / peroid)))
    return Signal(x, y, sig.params_hist)


def plot_sig(sig, title='', show=True):
    plt.plot(sig.x, sig.y)
    plt.xlabel('t[s]')
    plt.ylabel('A')
    plt.title(title)
    plt.xlim(0, sig.params_hist[0][1].duration)
    plt.savefig('img/' + title + '.png')
    if show:
        plt.show()
    plt.cla()
    plt.clf()


def plot_sig_lim(sig, title='', lim=10, show=True):
    plt.plot(sig.x, sig.y)
    plt.xlabel('t[s]')
    plt.ylabel('A')
    plt.title(title)
    plt.xlim(0, lim)
    plt.savefig(title + '.png')
    if show:
        plt.show()
    plt.cla()
    plt.clf()


def plot_quant_signals(sig1, sig2, sig3, sig4):
    plt.figure(1)
    plt.subplot(221)
    plot_sig(sig1, 'sygnal oryginalny', False)
    plt.subplot(222)
    plot_sig(sig2, 'sygnal sprobkowany', False)
    plt.subplot(223)
    plot_sig(sig3, 'sygnal skwantowany', False)
    plt.subplot(224)
    plot_sig(sig4, 'sygnal aproksymowany', False)
    plt.show()


def plot_hist(sig, binnum):
    sig = trunc(sig)
    if np.inf in sig.y:
        return
    plt.hist(sig.y, bins=binnum, edgecolor='black', linewidth=1.2)
    plt.show()


# ##################################################################
# ZAD 3
# ##################################################################


def convultion(sig1, sig2):
    params1 = sig1.params_hist[0][1]
    params2 = sig2.params_hist[0][1]
    length = len(sig1.x) + len(sig2.x) - 1
    x = np.arange(0, length * 1 / params1.sampfq, 1 / params1.sampfq)
    y = []

    for n in range(length):
        s = 0
        for k in range(len(sig1.y)):
            if 0 <= (n - k) < len(sig2.y):
                s += sig1.y[k] * sig2.y[n - k]
        y.append(s)

    params = copy.deepcopy(params1)
    params.duration = params1.duration + params2.duration
    return Signal(x, y, params)


def corelation(sig1, sig2):
    params1 = sig1.params_hist[0][1]
    params2 = sig2.params_hist[0][1]
    length = len(sig1.x) + len(sig2.x) - 1
    x = np.arange(0, length * 1 / params1.sampfq, 1 / params1.sampfq)
    y = []

    for n in range(-(len(sig2.x)), length - len(sig2.x), 1):
        s = 0
        for k in range(len(sig1.y)):
            if 0 <= (k - n) < len(sig2.y):
                s += sig1.y[k] * sig2.y[k - n]
        y.append(s)

    params = copy.deepcopy(params1)
    params.duration = params1.duration + params2.duration
    return Signal(x, y, params)


def impulse_ans(n, m, k):
    if n == (m - 1) / 2:
        return 2 / k
    else:
        return np.sin((2 * np.pi * (n - (m - 1) / 2)) / k) / (np.pi * (n - (m - 1) / 2))


def window_hanning(y, m):
    for i in range(len(y)):
        y[i] *= 0.5 - 0.5 * np.cos(2 * np.pi * i / m)
    return y


def window_rect(y, m):
    for i in range(len(y)):
        if (m - 1) / 2 < i < -(m - 1) / 2:
            y[i] = 0
    return y


def mid_filter(y):
    for i in range(len(y)):
        y[i] *= 2 * np.sin(np.pi * i / 2)
    return y


def gen_filter(n, m, fp, fo, rect=False, mid=False):
    k = fp / fo
    x = np.arange(0, 1 / fp * n, 1 / fp)
    y = []

    for i in range(n):
        y.append(impulse_ans(i, m, k))

    if mid:
        y = mid_filter(y)

    if rect:
        y = window_rect(y, m)
    else:
        y = window_hanning(y, m)

    return Signal(x, y, SignalParams(0, 0, 1 / fp * n, 0, 0, fp, None))


def reflect_sig(sig, v, s):
    t = s / v
    params = sig.params_hist[0][1]
    params_new = copy.deepcopy(params)
    params_new.offset = True
    params_new.time_start = t
    return gen_signal_from_params(params_new)


def calc_dist(sig, v, s):
    sig_ref = reflect_sig(sig, v, s)

    # korelacja przez splot:
    # sig_ref2 = copy.deepcopy(sig_ref)
    # sig_ref2.y = sig_ref2.y[::-1]
    # cor = convultion(sig, sig_ref2)

    cor = corelation(sig, sig_ref)

    # find max
    max_x = 0
    max_y = 0
    for i in range(len(cor.x)//2, len(cor.x), 1):
        if cor.y[i] > max_y:
            max_y = cor.y[i]
            max_x = cor.x[i]

    dt = max_x - cor.x[len(cor.x)//2]
    dist = v * dt
    ddist = abs(dist - s)

    print("Calulated dist:", dist)
    print("Delta dist:", ddist)

    plot_sig(sig, 'sent_sig', False)
    plot_sig(sig_ref, 'reflect_sig', False)
    plot_sig(cor, 'correlated_sig', False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', help='path to json config or saved signal', default='data/p1.json')
    parser.add_argument('--save', help='path to save signal', default='s0.bin')
    parser.add_argument('--savetxt', help='path to save signal in txt', default='s.txt')
    parser.add_argument('--bins', help='number of bins for histogram', type=int, default=10)
    parser.add_argument('--operation', help='optional operation [+-/x] with signals')
    parser.add_argument('--oppath', help='path to saved signal for operation')
    parser.add_argument('--spfq', help='sampling fq', type=int)
    parser.add_argument('--levels', help='quantization levels', type=int)
    parser.add_argument('--defq', help='dequant fq, default is old fq', type=int)

    results = parser.parse_args()
    readpath = results.read
    savepath = results.save
    savetxtpath = results.savetxt
    bins = results.bins
    sig1 = read(readpath)

    if results.operation:
        op = results.operation
        oppath = results.oppath
        sig2 = read(oppath)
        sig1 = sig_op(sig1, sig2, op)
    if savepath:
        to_bin(sig1, savepath)
    if savetxtpath:
        sig_to_txt(sig1, savetxtpath)

    # zad 2
    spfq = results.spfq
    levels = results.levels
    defq = 1 / (sig1.x[1] - sig1.x[0])
    if results.defq:
        defq = results.defq

    sig2 = sample(sig1, spfq)
    sig3 = quant(sig2, 0, levels - 1, True)
    sig4 = quant(sig3, min(sig1.y), max(sig1.y), False)
    sig5 = dequant_sinc(sig4, defq)

    print_stats_quant(sig5, sig1)
    plot_quant_signals(sig1, sig2, sig3, sig5)
    plot_hist(sig1, bins)


def conv_test():
    # splot
    params1 = SignalParams(3, 0, 5, 1, 0, 100, SignalType.SIN)
    params2 = SignalParams(3, 0, 10, 3, 0, 100, SignalType.SIN)
    # sig1 = Signal([0, 1, 2, 3], [1, 2, 3, 4], SignalParams(duration=4, time_start=0))
    # sig2 = Signal([0, 1, 2], [5, 6, 7], SignalParams(duration=4, time_start=0))
    sig1 = gen_signal_from_params(params1)
    sig2 = gen_signal_from_params(params2)
    sig3 = convultion(sig1, sig2)
    plot_sig(sig3, 'convultion', False)


def filter_test():
    fil = gen_filter(800, 7, 200, 20, rect=True, mid=True)

    params1 = SignalParams(2, 0, 4, 1, 0, 200, SignalType.SIN)
    params2 = SignalParams(0.1, 0, 4, 1, 0, 200, SignalType.NOISE_GAUSS)
    sig1 = gen_signal_from_params(params1)
    sig2 = gen_signal_from_params(params2)

    sig_noised = sig_add(sig1, sig2)
    sig_filtered = convultion(sig_noised, fil)

    plot_sig(fil, 'filter', False)
    plot_sig(sig_noised, 'sig_noised', False)
    plot_sig_lim(sig_filtered, 'filtered_signal', sig_noised.params_hist[0][1].duration, False)


def cor_test():
    # params1 = SignalParams(3, 0, 5, 1, 0, 100, SignalType.SIN)
    # params2 = SignalParams(3, 0, 10, 3, 0, 100, SignalType.SIN)
    sig1 = Signal([0, 1, 2, 3], [1, 2, 3, 4], SignalParams(duration=4, time_start=0, sampfq=1))
    sig2 = Signal([0, 1, 2], [5, 6, 7], SignalParams(duration=4, time_start=0, sampfq=1))
    # sig1 = gen_signal_from_params(params1)
    # sig2 = gen_signal_from_params(params2)
    sig3 = corelation(sig1, sig2)
    plot_sig(sig3, 'corelation')


def dist_test():
    params = SignalParams(2, 0, 4, 1, 0, 200, SignalType.SIN, False)
    sig = gen_signal_from_params(params)
    calc_dist(sig, 100, 50)


if __name__ == "__main__":
    dist_test()
