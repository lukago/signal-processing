import argparse
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
                 peroid=0.0, duty=0.0, sampfq=0.0, sigtype=None):
        self.amplitude = amplitude
        self.time_start = time_start
        self.duration = duration
        self.period = peroid
        self.duty = duty
        self.sampfq = sampfq
        self.type = sigtype


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
    plt.plot(sig.x, sig.y, marker='.')
    plt.xlabel('t[s]')
    plt.ylabel('A')
    plt.title(title)
    if show:
        plt.show()


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


if __name__ == "__main__":
    main()
