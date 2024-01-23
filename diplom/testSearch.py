import pandas as pd
import numpy as np
import operator
import time
import sys
import multiprocessing as mp
from math import log10
from sklearn.linear_model import Ridge
from io import StringIO
from csv import writer
from functools import partial
from sklearn.linear_model import LogisticRegression

QUERY_PATH = r'queries\thetaQ_hexapod_beg.h5'
THETA_PATH = r'theta_store.h5'
KOEFS_LOG = [1.00E+01, 1.50E+01, 2.50E+01, 5.00E+01,
         1.00E+02, 2.00E+02, 5.00E+02, 1.00E+03, 1.00E+05, 5.00E+05]
KOEFS_ADJ = [None, 0.1, 0.25, 0.5, 0.75, 0.1, 1, 1.25, 1.5, 1.75]
GLOB_REV = ['156579', '228969', '225845', '135529', '142740', '133843', '209062', '170711',
     '94012','81322','135570','96452','148964','123283','180571','174609','222535','5598',
     '149038','218379','222599','198456','118532','156027','155535','180167',
     '115334', '194068']

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('{} completed in {:.2f}s'.format(method.__name__, (te - ts)))
        return result
    return timed

def swaptheta(name, theta):
    thetaQ = pd.read_hdf(r'queries\thetaQ_'+name+'.h5','table')
    thetaQ = thetaQ[[col for col in thetaQ.columns[50:]]]
    mera = cosin(theta, thetaQ)
    return thetaQ, mera

def cosin(theta, thetaQ_in):
    thetaQ = pd.DataFrame(thetaQ_in)
    mera = {}
    start = time.time()
    meraDOWN1= pow(pow(thetaQ.values[0], 2).sum(), 1/2)
    for row in range(len(theta.index)):
        meraUP = thetaQ.values[0].dot(theta.values[row])
        if meraUP == 0:
            continue
        meraDOWN2 = pow(pow(theta.values[row], 2).sum(), 1/2)
        mera[theta.index[row]] = meraUP/(meraDOWN1*meraDOWN2)

    mera = sorted(mera.items(), key=operator.itemgetter(1))
    mera.reverse()
    end = time.time() - start
    print('Calculate in {0:.2f}s...'.format(end))
    return mera

######### adjust ############# start
def adjust_query(rev, irrev,theta, thetaQ, koef=None):
    nrev = len(rev)
    nirrev = len(irrev)
    if koef is None:
        if nrev*nirrev != 0:
            koef = nrev/nirrev if nirrev > nrev else nirrev/nrev
        else:
            koef = 1
    n = nrev + nirrev
    Qorig = pd.Series(data=thetaQ.values[0],index=thetaQ.columns)
    Qrev = pd.Series(data=np.zeros(thetaQ.shape[1]),index=thetaQ.columns)
    Qirrev = pd.Series(data=np.zeros(thetaQ.shape[1]),index=thetaQ.columns)
    for relev in rev:
        Qrev = Qrev.add(theta.loc[relev]).multiply(koef*nirrev/n)
    for irrelev in irrev:
        Qirrev = Qirrev.add(theta.loc[irrelev]).multiply(koef*nrev/n)
    res = Qorig.add(Qrev).subtract(Qirrev)
    res = (res - res.min()) / (res.max() - res.min())
    res = res.divide(res.sum())
    return pd.DataFrame(res).transpose()

def muda(mera_glob, theta_glob, thetaQ_glob, rev, irrev, koef):
    theta = theta_glob.loc[[item[0] for item in mera_glob]]
    thetaQ = adjust_query(rev, irrev, theta_glob, thetaQ_glob, koef)
    mera = cosin(theta, thetaQ)
    docs = 20
    printed = print_adjust(mera, rev, irrev, thetaQ_glob.index[0])
    return printed

def print_adjust(mera, rev, irrev, doc_name, entries=20):
    to_print = []
    for item in mera:
        if len(to_print) == entries:
            break
        if item[0] not in rev and item[0] not in irrev:
            to_print.append(item)
    print('\nTop {} docs with max cosin for {}:'.format(entries, doc_name))
    print(*to_print, sep='\n')
    return [item[0] for item in to_print]

def get_adjust_irrev(printed, GLOB_REV, frequency=1):
    irrev = []
    for item in printed:
        if item not in GLOB_REV:
            irrev.append(item)
    return irrev[::frequency]

def check_adjust(printed, GLOB_REV, rev, irrev):
    _rev = []
    for item in printed:
        if item in GLOB_REV:
             _rev.append(item)
    print('\nRelevant docs here [{}]:'.format(len(_rev)))
    print(*_rev, sep='\n')
    print('\nrev/irrev - {}/{}'.format(len(rev),len(irrev)))
    return _rev

def iterate_adjust(theta, thetaQ, mera100, rev, irrev, GLOB_REV, koef, frequencyIREV, frequencyREV):
    printed = muda(mera100, theta, thetaQ, rev, irrev, koef)
    new_rev = check_adjust(printed, GLOB_REV, rev, irrev)
    rev.extend(new_rev[::frequencyREV])
    irrev.extend(get_adjust_irrev(printed, GLOB_REV, frequencyIREV))
    return rev, irrev

def adjust_forme(theta, thetaQ, mera, koefs, GLOB_REV, frequencyIREV, frequencyREV):
    steps = {}
    for koef in koefs:
        rev, irrev = [], []
        steps[koef if koef is not None else 'default'] = []
        while(len(rev) < len(GLOB_REV)):
              rev, irrev = iterate_adjust(theta, thetaQ, mera, rev, irrev, GLOB_REV, koef, frequencyIREV, frequencyREV)
              steps[koef if koef is not None else 'default'].append((len(rev), len(irrev)))
    print('\n###### QEURY ADJUSTS RESULTS ######')
    for key in steps:
        print('\nFor koef {} ({} iterations):'.format(key, len(steps[key])))
        for item in steps[key]:
              print('{}/{}'.format(item[0], item[1]))
######### adjust ############# end

######### logistic ############# start
def data_logistic(theta, mera, rev, irrev, confidence=1e15):
    items = len(mera)
    _x = np.ones((items, 200))
    _y = np.ones(items)
    weights = np.ones(items)
    n = (len(rev) + len(irrev))
    koef_rev = len(irrev) / n
    koef_irrev = len(rev) / n
    for i, item in enumerate(mera):
        _x[i] = theta.loc[item[0]].values
        if item[0] in rev:
            _y[i] = 1
            weights[i] = confidence * koef_rev
        elif item[0] in irrev:
            _y[i] = 0
            weights[i] = confidence * koef_irrev
    return _x, _y, weights

def logistic(theta, mera, rev, irrev, confidence):
    _x, _y, weights = data_logistic(theta, mera, rev, irrev, confidence)
    fitting_index = weights > 1
    logistic = LogisticRegression()
    logistic.fit(_x[fitting_index], _y[fitting_index], weights[fitting_index])
    return logistic.predict_proba(_x), _x, _y, weights

def print_logistic(predicted, mera, rev, irrev, GLOB_REV, entries=20, forme=True):
    new_mera = [x for (x, y) in sorted(zip(mera, predicted),
                                       key=lambda pair: pair[1][1], reverse=True)]
    to_print = []
    for item in new_mera:
        if item[0] in rev or item[0] in irrev:
            continue
        if len(to_print) == entries:
            break
        to_print.append(item)

    new_rev = []
    for item in to_print:
        if item[0] in GLOB_REV:
            new_rev.append(item)
    if not forme:
        print(*to_print, sep='\n')
        print('\nNew relevant documents here [{}]:'.format(len(new_rev)))
        print(*new_rev, sep='\n')
        print('rev/irrev - [{}/{}]'.format(len(rev), len(irrev)))
    return to_print, new_rev

def first_logistic(mera, GLOB_REV, rev_n, entries, frequency):
    rev = GLOB_REV[:rev_n]
    irrev = get_irrev_logistic(mera[:entries], rev, [], frequency)
    return rev, irrev

def iterate_logistic(theta, mera, rev, irrev, frequency, confidence, entries, forme=False):
    predicted, x, y, w = logistic(theta, mera, rev, irrev, confidence)
    # check_logistic(c, y, mera, GLOB_REV, detailed)
    printed, new_rev = print_logistic(predicted, mera, rev, irrev, GLOB_REV, entries, forme)
    rev.extend([item[0] for item in new_rev])
    if forme:
        print(len(new_rev), end=' ')
    irrev = get_irrev_logistic(printed, rev, irrev, frequency)
    return rev, irrev

def get_irrev_logistic(printed, rev, irrev, frequency=1):
    new_irrev = []
    for item in printed:
        if item[0] not in rev:
            new_irrev.append(item[0])
    irrev.extend(new_irrev[::frequency])
    return irrev

def check_logistic(c, y, mera, GLOB_REV, detailed=False):
    buf1 = []
    buf2 = {'in c, not in glob': [],
            'in glob not in c': [],
            'in c and glob not in rev': []}
    for item in c.nonzero()[0]:
        if item not in y.nonzero()[0]:
            buf1.append(mera[item])
            if mera[item][0] in GLOB_REV:
                buf2['in c and glob not in rev'].append(mera[item])
        if mera[item][0] not in GLOB_REV:
            buf2['in c, not in glob'].append(mera[item])
    print('Items in c but not in y [{}]:'.format(len(buf1)))
    if detailed: print(*buf1, sep='\n')

    buf3 = []
    for item in y.nonzero()[0]:
        if item not in c.nonzero()[0]:
            buf3.append(mera[item])
    print('\nItems in y but not in c (forgotten) [{}]:'.format(len(buf3)))
    if detailed: print(*buf3, sep='\n')

    for item in GLOB_REV:
        if item not in list(map(lambda x: mera[x][0], c.nonzero()[0])):
            buf2['in glob not in c'].append(item)

    print('\nItems in c but not in glob (incorrectly predicted) [{}]:'.format(len(buf2['in c, not in glob'])))
    if detailed: print(*buf2['in c, not in glob'], sep='\n')
    print('\nItems in glob but not in c (not predicted) [{}]:'.format(len(buf2['in glob not in c'])))
    if detailed: print(*buf2['in glob not in c'], sep='\n')
    print(
        '\nItems in c and glob but not in y (correctly predicted) [{}]:'.format(len(buf2['in c and glob not in rev'])))
    if detailed: print(*buf2['in c and glob not in rev'], sep='\n')
    print('\nItems in c - [{}]'.format(c.nonzero()[0].size))

def log_forme(theta, koefs, mera, GLOB_REV, n_rev, entries, frequency, no_det=True):
    print('\n###### LOG ADJUSTS RESULTS ######')
    for koef in koefs:
        koef_str = '{:E}'.format(koef)
        koef_str = koef_str.split('E')[0].rstrip('0').rstrip('.') + 'e' + koef_str.split('E')[1]
        print('For koef [{}] :'.format(koef_str))
        rev, irrev = first_logistic(mera, GLOB_REV, n_rev, entries, frequency)
        while len(rev) < len(GLOB_REV):
            rev, irrev = iterate_logistic(theta, mera, rev, irrev, frequency, koef, entries, no_det)
        print('\n')
######### logistic ############# end|

def usual_check(mera, GLOB_REV):
    print('\nCosine measure distribution:')
    for i in range(0,100,20):
        cnt = 0
        for item in mera[i:i+20]:
            if item[0] in GLOB_REV:
                cnt += 1
        print(cnt)

def main():
    start_glob = time.time()
    thetaQ = pd.read_hdf(QUERY_PATH,'table')
    theta = pd.read_hdf(THETA_PATH,'table')
    theta = theta[[col for col in theta.columns[50:]]]
    thetaQ = thetaQ[[col for col in thetaQ.columns[50:]]]
    end = time.time() - start_glob
    print('Read matrices in {0:.2f}s...'.format(end))

    mera = cosin(theta, thetaQ)
    adjust_forme(theta, thetaQ, mera[:100], KOEFS_ADJ, GLOB_REV, 1, 1)
    log_forme(theta, KOEFS_LOG, mera[:100], GLOB_REV, 6, 20, 1)

if __name__ == '__main__':
    main()
