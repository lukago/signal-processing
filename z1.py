from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import json


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


def to_json(params, filename):
    with open(filename, 'w') as outfile:
        json.dump(params.__dict__, outfile, indent=2)


def from_json(filename):
    with open(filename, 'r') as infile:
        js = infile.read()
        return json.loads(js)


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


def main():
    p1 = SignalParams(3, 10, 10, 3, 0, 0.01, SignalType.SIN)
    p2 = SignalParams(3, 10, 10, 3, 0, 0.01, SignalType.SIN_SINGLE)
    p3 = SignalParams(3, 10, 10, 3, 0, 0.01, SignalType.SIN_DOUBLE)
    p4 = SignalParams(1, 16, 10, 3, 0.5, 0.01, SignalType.RECT)
    p5 = SignalParams(1, 16, 10, 3, 0.5, 0.01, SignalType.RECT_SIMETRIC)
    p6 = SignalParams(1, 12, 10, 4, 0.9, 0.01, SignalType.TRI)
    p7 = SignalParams(1, 12, 10, 4, 0.5, 0.01, SignalType.STEP)
    p8 = SignalParams(1, 2, 4, 2, 0.0, 0.01, SignalType.UNIT_IMPULSE)
    p9 = SignalParams(1, 12, 10, 4, 0.0, 0.01, SignalType.NOISE_GAUSS)
    p10 = SignalParams(1, 12, 10, 4, 0.01, 0.01, SignalType.NOISE_IMPULSE)
    p11 = SignalParams(1, 12, 10, 4, 0.0, 0.01, SignalType.NOISE_NORM)
    to_json(p1, 'b.json')
    px = from_json('b.json')
    print(px)

    sig1 = sig_sin(p1)
    sig2 = sig_sin_single(p2)
    sig3 = sig_sin_double(p3)
    sig4 = sig_rect(p4)
    sig5 = sig_rect(p5)
    sig6 = sig_tri(p6)
    sig7 = sig_step(p7)
    sig8 = sig_impulse(p8)
    sig9 = noise_gauss(p9)
    sig10 = noise_impulse(p10)
    sig11 = noise_normal(p11)

    print_stats(sig1)
    plot_hist(sig1)
    plot_sig(sig1)


if __name__ == "__main__":
    main()
