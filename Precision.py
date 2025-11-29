#!/usr/bin/env python3
# Compute DG/Kubilius hybrid approximation for M(q) (probability TF contains prime q).
# Stable (no overflow) version. Preset for q=2 with ~8-digit target.
#
# Usage: run with Python 3.8+ (single-core). Increasing P_MAX increases precision.

import math, time
from functools import lru_cache

# ---------------------------
# PARAMETERS (tweak if needed)
# ---------------------------
P_MAX = 5_000_000   # prime cutoff. Increase for smaller error; ~350k primes here.
E_MAX = 80          # exponent cutoff (exactly computed for e <= E_MAX)
q = 2               # target prime
# conservative multiplicative tail factor used in prime-tail estimate
PRIME_TAIL_FACTOR = 2.0

# ---------------------------
# Sieve primes up to P_MAX (memory efficient)
# ---------------------------
def primes_upto(N):
    sieve = bytearray(b'\x01') * (N+1)
    sieve[0:2] = b'\x00\x00'
    lim = int(N**0.5) + 1
    for p in range(2, lim):
        if sieve[p]:
            step = p
            start = p*p
            sieve[start:N+1:step] = b'\x00' * (((N - start)//step)+1)
    return [i for i, flag in enumerate(sieve) if flag]

# ---------------------------
# tf_contains_q recursion for small exponents (deterministic)
# ---------------------------
@lru_cache(None)
def tf_contains_q(m, q):
    # returns True iff q appears anywhere in the tower of the integer m
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
        if temp == q:
            return True
    return False

# ---------------------------
# small-Omega function for small integers
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
# Main driver
# ---------------------------
def compute_Mq(P_MAX, E_MAX, q, prime_tail_factor=2.0):
    t0 = time.time()
    primes = primes_upto(P_MAX)
    t1 = time.time()

    # precompute w(e) for e <= E_MAX
    w = [0]*(E_MAX+1)
    for e in range(E_MAX+1):
        w[e] = 0 if tf_contains_q(e, q) else 1

    # precompute Omega up to a moderate tail (only needed if we used alpha^Omega)
    # we won't rely on alpha^Omega for rigorous bound; but having Omega is cheap for small e.
    e_tail_max = max(E_MAX+50, 200)
    omega_cache = {e: Omega(e) for e in range(E_MAX+1)}
    for e in range(E_MAX+1, e_tail_max+1):
        omega_cache[e] = Omega(e)

    # compute log product over primes <= P_MAX; stable via inverse powers
    log_prod = 0.0
    for p in primes:
        one_minus = 1.0 - 1.0/p
        # e=0 term
        s = one_minus  # (1-1/p) * 1
        if p == q:
            # only e=0 contributes for p==q
            log_prod += math.log(s)
            continue
        inv_p = 1.0 / p
        inv_p_pow = inv_p  # equals p^{-1}
        # exact small exponents e=1..E_MAX using w(e)
        for e in range(1, E_MAX+1):
            if w[e]:
                s += one_minus * inv_p_pow
            inv_p_pow *= inv_p
        # for rigorous upper bound on omitted exponent tail we can add
        # (1-1/p) * sum_{e>E_MAX} p^{-e} = p^{-(E_MAX+1)}
        tail_bound_per_prime = inv_p_pow / (1.0 - inv_p)  # equals sum_{e=E_MAX+1..inf} p^{-e}
        s += one_minus * tail_bound_per_prime
        # ensure s positive (it should be)
        if s <= 0.0:
            s = 1e-300
        log_prod += math.log(s)

    alpha_trunc = math.exp(log_prod)
    M_trunc = 1.0 - alpha_trunc
    t2 = time.time()

    # ---------------------------
    # error estimates
    # ---------------------------
    # exponent tail (crude but tiny): <= sum_{n>=2} n^{-(E_MAX+1)} <= 2^{-(E_MAX+1)} + ...
    exp_tail_bound = 2.0**(-(E_MAX+1))  # extremely small for E_MAX>=50

    # prime tail: approximate sum_{p>P} 1/p^2 approx ∫ (1/(t^2 log t)) dt ~ 1/(P log P)
    # use conservative multiplicative factor
    P = P_MAX
    prime_tail_bound = prime_tail_factor / (P * math.log(P))  # conservative

    # combine absolute error bound (conservative)
    total_abs_error = exp_tail_bound + prime_tail_bound

    lower = max(0.0, M_trunc - total_abs_error)
    upper = min(1.0, M_trunc + total_abs_error)

    t3 = time.time()
    return {
        'P_MAX': P_MAX, 'E_MAX': E_MAX, 'num_primes': len(primes),
        'M_trunc': M_trunc, 'alpha_trunc': alpha_trunc,
        'exp_tail_bound': exp_tail_bound, 'prime_tail_bound': prime_tail_bound,
        'total_abs_error': total_abs_error, 'interval': (lower, upper),
        'time_sieve': t1-t0, 'time_prod': t2-t1, 'time_total': t3-t0
    }

# ---------------------------
# Run and report
# ---------------------------
if __name__ == "__main__":
    t_start = time.time()
    out = compute_Mq(P_MAX=P_MAX, E_MAX=E_MAX, q=q, prime_tail_factor=PRIME_TAIL_FACTOR)
    print(f"Settings: P_MAX={out['P_MAX']:,}, E_MAX={out['E_MAX']}, target q={q}")
    print(f"Primes used: {out['num_primes']:,}")
    print(f"Sieve time: {out['time_sieve']:.2f}s, product time: {out['time_prod']:.2f}s, total: {out['time_total']:.2f}s")
    print()
    print(f"Truncated (computed) value: M({q}) ≈ {out['M_trunc']:.12f}")
    print(f"Estimated exponent-tail bound <= {out['exp_tail_bound']:.2e}")
    print(f"Estimated prime-tail bound <= {out['prime_tail_bound']:.2e} (conservative)")
    print(f"Combined absolute error bound <= {out['total_abs_error']:.2e}")
    low, high = out['interval']
    print(f"Conservative interval: M({q}) ∈ [{low:.12f}, {high:.12f}]")
    print(f"Interval width ≈ {high-low:.2e}")
    # suggestion:
    if out['total_abs_error'] > 1e-8:
        print("\nNote: conservative error bound is > 1e-8. To reduce it, increase P_MAX.")
    else:
        print("\nTarget: conservative error bound <= 1e-8 — achieved with these parameters.")
