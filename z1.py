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
    def __init__(self, amplitude=0.0, time_start=0.0, duration=0.0, peroid=0.0, duty=0.0, step=0.0, sigtype=None):
        self.amplitude = amplitude
        self.time_start = time_start
        self.duration = duration
        self.period = peroid
        self.duty = duty
        self.step = step
        self.type = sigtype


def to_json(pyobj, filename):
    with open(filename, 'w') as outfile:
        json.dump(pyobj.__dict__, outfile, indent=2)


def to_bin(pyobj, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(pyobj, outfile)


def from_json(filename):
    with open(filename, 'r') as infile:
        js = infile.read()
        return json.loads(js)


def from_bin(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def sig_sin(params):
    x = np.arange(params.time_start, params.duration + params.time_start, params.step)
    y = params.amplitude * np.sin(2 * np.pi / params.period * x)
    return {'x': x, 'y': y}


def sig_sin_single(params):
    x = np.arange(params.time_start, params.duration + params.time_start, params.step)
    y = params.amplitude * np.sin(2 * np.pi / params.period * x)
    y[y < 0] = 0
    return {'x': x, 'y': y}


def sig_sin_double(params):
    x = np.arange(params.time_start, params.duration + params.time_start, params.step)
    y = params.amplitude * np.sin(2 * np.pi / params.period * x)
    y = abs(y)
    return {'x': x, 'y': y}


def sig_tri(params):
    x = np.arange(params.time_start, params.duration + params.time_start, params.step)
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
    return {'x': x, 'y': y}


def sig_rect(params):
    x = np.arange(params.time_start, params.duration + params.time_start, params.step)
    y = []
    cnt = 0
    for i in range(len(x)):
        if x[i] > params.period * cnt + params.period + params.time_start:
            cnt += 1
        if x[i] < params.period * params.duty + cnt * params.period + params.time_start:
            y.append(params.amplitude)
        else:
            y.append(0) if params.type == SignalType.RECT else y.append(-params.amplitude)
    return {'x': x, 'y': y}


def sig_step(params):
    x = np.arange(params.time_start, params.duration + params.time_start, params.step)
    y = []
    for i in range(len(x)):
        if x[i] < params.period + params.time_start:
            y.append(0)
        elif x[i] == params.period + params.time_start:
            y.append(params.amplitude / 2)
        else:
            y.append(params.amplitude)
    return {'x': x, 'y': y}


def sig_impulse(params):
    eps = 0.00001
    x = np.arange(params.time_start, params.duration + params.time_start, params.step)
    y = []
    for i in range(len(x)):
        if abs(x[i] - params.period - params.time_start) < eps:
            y.append(params.amplitude)
        else:
            y.append(0)
    return {'x': x, 'y': y}


def noise_normal(params):
    x = np.arange(params.time_start, params.duration + params.time_start, params.step)
    y = np.random.uniform(-params.amplitude, params.amplitude, len(x))
    return {'x': x, 'y': y}


def noise_gauss(params):
    x = np.arange(params.time_start, params.duration + params.time_start, params.step)
    y = np.random.normal(0, params.amplitude, len(x))
    return {'x': x, 'y': y}


def noise_impulse(params):
    x = np.arange(params.time_start, params.duration + params.time_start, params.step)
    y = []
    for i in range(len(x)):
        if np.random.random() < params.duty:
            y.append(params.amplitude)
        else:
            y.append(0)
    return {'x': x, 'y': y}


def sig_add(sig1, sig2):
    sig1['y'] = np.add(sig1['y'], sig2['y'])
    return sig1


def sig_sub(sig1, sig2):
    sig1['y'] = np.subtract(sig1['y'], sig2['y'])
    return sig1


def sig_mul(sig1, sig2):
    sig1['y'] = np.multiply(sig1['y'], sig2['y'])
    return sig1


def sig_div(sig1, sig2):
    eps = 0.00001
    y1 = sig1['y']
    y2 = sig2['y']
    for i in range(len(y1)):
        if abs(y2[i]) < eps:
            y1[i] = np.inf if y1[i] > 0 else -np.inf
        else:
            y1[i] /= y2[i]

    return {'x': sig1['x'], 'y': y1}


def avg(sig):
    return np.sum(sig['y']) / len(sig['y'])


def avg_abs(sig):
    y = abs(sig['y'])
    return np.sum(y / len(sig['y']))


def rms(sig):
    return np.sqrt(avg_pow(sig))


def variance(sig):
    x = avg(sig)
    return sum((y - x) ** 2 for y in sig['y'])


def avg_pow(sig):
    return sum(y * y for y in sig['y']) / len(sig['y'])


def plot_sig(signal):
    plt.plot(signal['x'], signal['y'])
    plt.xlabel('t[s]')
    plt.ylabel('A')
    plt.show()


def plot_hist(signal):
    plt.hist(signal['y'], edgecolor='black', linewidth=1.2)
    plt.show()


def print_stats(sig):
    print('Average value: ', avg(sig))
    print('Average absolute value: ', avg_abs(sig))
    print('Root mean square: ', rms(sig))
    print('Variance:', variance(sig))
    print('Average power: ', avg_pow(sig))


def read(path):
    words = path.split('.')
    ext = words.pop()
    if ext == 'json':
        params = from_json(path)
        return gen_signal(params)
    elif ext == 'bin':
        return from_bin(path)
    else:
        return from_json(path)


def gen_signal(p):
    params = SignalParams(p['amplitude'], p['time_start'], p['duration'], p['period'], p['duty'], p['step'], p['type'])
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
    if op == '*':
        return sig_mul(sig1, sig2)
    if op == '/':
        return sig_div(sig1, sig2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', help='path to json config or saved signal', default='data/p1.json')
    parser.add_argument('--save', help='path to save signal', default='s.bin')
    parser.add_argument('--operation', help='operation [+-/*] and path to saved signal', default='*')
    parser.add_argument('--oppath', help='path to saved signal for operation', default='data/p9.json')

    results = parser.parse_args()
    readpath = results.read
    savepath = results.save

    sig1 = read(readpath)
    if results.operation:
        op = results.operation
        oppath = results.oppath
        sig2 = read(oppath)
        sig1 = sig_op(sig1, sig2, op)
    if savepath:
        to_bin(sig1, savepath)

    print_stats(sig1)
    plot_sig(sig1)
    plot_hist(sig1)


if __name__ == "__main__":
    main()
