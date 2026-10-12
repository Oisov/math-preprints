"""Monte Carlo part of the certificate (direct formula of Theorem 'norm', reference t=0).

Per sample: g1, g2 ~ N(0, I_240) (numpy PCG64, seed sequence [SEED, batch]),
D = Ld g1 + i Ld g2 (D_ref = 0), uniform anchor j, R_i = sech((t_i-t_j)/2) e^{D_i}
(common factors dropped), f = m|R^T B R|/|R|^2, c = m R^* B R/|R|^2, X = f - c.
Records exactly rounded sums (math.fsum per batch, Fractions across batches) and the
per-run maxima needed by the floating-point error budget.

Usage: python sample.py SEED NBATCHES [BATCH=20000] [PROCS=16]  -> samples_SEED_NBATCHES.json
"""
import sys, os, json, math, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from fractions import Fraction as Fr
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    cand = json.load(open(os.path.join(HERE, 'candidate.json')))
    V = np.array([[float.fromhex(x) for x in row] for row in cand['V']])
    A = np.array([[float.fromhex(x) for x in row] for row in cand['A']])
    return V, A, np.load(os.path.join(HERE, 'Ld.npy')), np.load(os.path.join(HERE, 'ell.npy'))


def batch(args):
    seed, b, nb, V, A, Ld, ell = args
    m = V.shape[0]; ref = m // 2
    rng = np.random.default_rng([seed, b])
    g1 = rng.standard_normal((nb, m - 1))
    g2 = rng.standard_normal((nb, m - 1))
    j = rng.integers(0, m, nb)
    gn2 = (g1 * g1).sum(1) + (g2 * g2).sum(1)
    ReD = np.zeros((nb, m)); ImD = np.zeros((nb, m))
    ReD[:, np.r_[0:ref, ref + 1:m]] = g1 @ Ld.T
    ImD[:, np.r_[0:ref, ref + 1:m]] = g2 @ Ld.T
    w = ReD - ell[np.arange(m)[None, :] - j[:, None] + (m - 1)]
    w = w - w.max(1, keepdims=True)
    mod = np.exp(w)
    Rr = mod * np.cos(ImD); Ri = mod * np.sin(ImD)
    S = (Rr * Rr + Ri * Ri).sum(1)
    Wr = Rr @ V; Wi = Ri @ V
    q_rr = np.einsum('na,ab,nb->n', Wr, A, Wr)
    q_ii = np.einsum('na,ab,nb->n', Wi, A, Wi)
    q_ri = np.einsum('na,ab,nb->n', Wr, A, Wi)
    f = m * np.hypot(q_rr - q_ii, 2 * q_ri) / S
    c = m * (q_rr + q_ii) / S
    X = f - c
    return dict(n=nb, sX=math.fsum(X), sX2=math.fsum(X * X), sabsX=math.fsum(np.abs(X)),
                sf=math.fsum(f), sc=math.fsum(c),
                max_gnorm2=float(gn2.max()), max_absReD=float(np.abs(ReD).max()),
                max_absX=float(np.abs(X).max()), max_f=float(f.max()))


def main():
    seed = int(sys.argv[1]); nbat = int(sys.argv[2])
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    V, A, Ld, ell = load()
    t0 = time.time()
    with Pool(procs) as p:
        res = p.map(batch, [(seed, b, nb, V, A, Ld, ell) for b in range(nbat)], chunksize=1)
    out = dict(seed=seed, nbatches=nbat, batch=nb, n=sum(r['n'] for r in res))
    for key in ('sX', 'sX2', 'sabsX', 'sf', 'sc'):
        s = sum((Fr(r[key]) for r in res), Fr(0))
        out[key] = [s.numerator, s.denominator]
        # each batch sum is the exactly rounded value of the true float sum: |err| <= 2^-53 |sum|
        out[key + '_err'] = float(sum(abs(Fr(r[key])) for r in res) * Fr(1, 2 ** 52))
    for key in ('max_gnorm2', 'max_absReD', 'max_absX', 'max_f'):
        out[key] = max(r[key] for r in res)
    out['secs'] = time.time() - t0
    n = out['n']
    print('n', n, 'mean X', out['sX'][0] / out['sX'][1] / n, 'mean f', out['sf'][0] / out['sf'][1] / n,
          'mean c', out['sc'][0] / out['sc'][1] / n, 'max f', out['max_f'], 'max|X|', out['max_absX'],
          'secs', round(out['secs'], 1))
    json.dump(out, open(os.path.join(HERE, f'samples_{seed}_{nbat}.json'), 'w'))


if __name__ == '__main__':
    main()
