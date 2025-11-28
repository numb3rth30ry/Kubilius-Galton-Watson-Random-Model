# Complete script to compute M(q) for q = 2,3,5,7,11
# (Kubilius-style fixed-point with hybrid exact-tail strategy)
import math
from functools import lru_cache

# ---------------------------
# Utility: primes up to N
# ---------------------------
def primes_upto(N):
    sieve = bytearray(b'\x01')*(N+1)
    sieve[0:2] = b'\x00\x00'
    for p in range(2, int(N**0.5)+1):
        if sieve[p]:
            step = p
            start = p*p
            sieve[start:N+1:step] = b'\x00' * (((N - start)//step)+1)
    return [i for i, isprime in enumerate(sieve) if isprime]

# ---------------------------
# Omega: total number of prime factors with multiplicity
# ---------------------------
def Omega(n):
    if n <= 1:
        return 0
    cnt = 0
    temp = n
    p = 2
    while p*p <= temp:
        while temp % p == 0:
            cnt += 1
            temp //= p
        p += 1 if p==2 else 2
    if temp > 1:
        cnt += 1
    return cnt

# ---------------------------
# tf_contains_q: recursion to check if q appears in TF(m)
# ---------------------------
@lru_cache(None)
def tf_contains_q(m, q):
    # q appears in TF(m) iff:
    #  - q divides m (base-level), or
    #  - some exponent in m's factorization contains q recursively
    if m <= 1:
        return False
    temp = m
    p = 2
    while p*p <= temp:
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                e += 1
                temp //= p
            if p == q:
                return True
            if tf_contains_q(e, q):
                return True
        p += 1 if p==2 else 2
    if temp > 1:
        # temp is prime
        if temp == q:
            return True
        # exponent on last prime is 1, and tf_contains_q(1,q) is False
    return False

# ---------------------------
# precompute w(e) = 1 if q NOT in TF(e), else 0
# for e in [0..E_MAX]
# ---------------------------
def precompute_w(E_MAX, q):
    w = [None]*(E_MAX+1)
    for e in range(E_MAX+1):
        w[e] = 0 if tf_contains_q(e, q) else 1
    return w

# ---------------------------
# main fixed-point solver (the code you pasted, completed)
# ---------------------------
def compute_alpha_for_q_corrected(q, primes, E_MAX, tol=1e-11, maxiter=200):
    # precompute small-exponent indicators
    w = precompute_w(E_MAX, q)

    # global omega cache (small)
    omega_cache = {e: Omega(e) for e in range(E_MAX+1)}

    def inner_sum(p, alpha):
        s = 0.0
        one_minus_1_over_p = 1.0 - 1.0/p
        # e=0 term
        s += one_minus_1_over_p * 1.0  # w(0)=1
        # e from 1..E_MAX exact w(e) except when p==q then w(e)=0
        p_pow = p
        if p == q:
            # only e=0 contributes: already added; remaining terms are zero
            return s
        for e in range(1, E_MAX+1):
            term = one_minus_1_over_p * (1.0 / p_pow) * w[e]
            s += term
            p_pow *= p
        # tail e > E_MAX: approximate using alpha^{Omega(e)} (p != q here)
        e_tail_max = 200
        # continue p_pow from p^{E_MAX+1}
        for e in range(E_MAX+1, e_tail_max+1):
            om = omega_cache.get(e)
            if om is None:
                om = Omega(e)
                omega_cache[e] = om
            term = one_minus_1_over_p * (1.0 / p_pow) * (alpha**om)
            s += term
            p_pow *= p
            if term < 1e-18:
                break
        return s

    alpha = 1.0
    for it in range(maxiter):
        log_prod = 0.0
        for p in primes:
            val = inner_sum(p, alpha)
            if val <= 0:
                val = 1e-300
            log_prod += math.log(val)
        alpha_new = math.exp(log_prod)
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new
    Mq = 1.0 - alpha
    return alpha, Mq, it+1

# ---------------------------
# Parameters (match the run that produced the numbers)
# ---------------------------
P_MAX = 10000       # primes up to 10k
primes = primes_upto(P_MAX)
E_MAX = 50          # compute exact w(e) for e <= 50
tol = 1e-11
maxiter = 200
qs = [2,3,5,7,11]

# ---------------------------
# Run for the requested primes
# ---------------------------
results_corrected = {}
for q in qs:
    alpha, Mq, iterations = compute_alpha_for_q_corrected(q, primes, E_MAX, tol=tol, maxiter=maxiter)
    results_corrected[q] = {'alpha': alpha, 'M': Mq, 'iters': iterations}

# print nicely
for q in qs:
    r = results_corrected[q]
    print(f"q={q:2d}  alpha={r['alpha']:.15f}  M={r['M']:.15f}  iters={r['iters']}")

# If you want rounded to 5 decimals:
print("\nRounded to 5 decimals:")
for q in qs:
    print(f" M({q}) = {results_corrected[q]['M']:.5f}")
