# Reproduce the Kubilius product evaluation for tower-height probabilities.
# Pure Python, adjustable cutoffs.
import math
from itertools import compress

def primes_upto(N):
    sieve = bytearray(b'\x01')*(N+1)
    sieve[0:2] = b'\x00\x00'
    for p in range(2, int(N**0.5)+1):
        if sieve[p]:
            step = p
            start = p*p
            sieve[start:N+1:step] = b'\x00' * (((N - start)//step) + 1)
    return [i for i, isprime in enumerate(sieve) if isprime]

# compute h_int(m) up to K_max by factorization
def factorize_small(n, primes):
    res = {}
    for p in primes:
        if p*p > n: break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p; e += 1
            res[p] = e
    if n > 1:
        res[n] = 1
    return res

def compute_h_int(K_max, primes):
    h = {0:0, 1:0}   # <-- FIXED: include 0
    for m in range(2, K_max+1):
        f = factorize_small(m, primes)
        if not f:
            h[m] = 1   # m is prime
        else:
            max_child = 0
            for v in f.values():
                max_child = max(max_child, h.get(v, 0))
            h[m] = 1 + max_child
    return h

def r_p_of_t(p, t, h_int, k_trunc=100):
    # r_p(t) = sum_{k>=0, h_int(k)<=t} (1-1/p) p^{-k}
    # We sum k from 0..k_trunc; for k>len(h_int) treat h_int(k) by factorization fallback (very small prob).
    one_minus = 1.0 - 1.0/p
    s = 0.0
    for k in range(0, k_trunc+1):
        # h_int for k: if k==0 -> 0; else use dictionary or approximate
        hk = h_int.get(k)
        if hk is None:
            # if k > precomputed, approximate: for large k, h_int(k) will be >= 1 and tails p^{-k} tiny
            # conservatively treat hk large (so condition likely false); here just break when p^{-k} negligible
            if p**(-k) < 1e-16:
                break
            hk = 999
        if hk <= t:
            s += one_minus * (p**(-k))
    return s

def compute_probabilities(prime_cutoff=200000, K_max=200, k_trunc=80, prob_levels=6):
    primes = primes_upto(min(prime_cutoff, 200000))  # adjust sieve limit as needed
    h_int = compute_h_int(K_max, primes)
    # if you want more primes, regenerate primes with larger limit
    selected_primes = primes  # can be truncated if you want fewer primes for speed
    # compute r_p for each level t = 0..prob_levels-1 (we need r_p(H-1) for H up to prob_levels)
    r_by_p_and_t = {}
    for p in selected_primes:
        r_by_p_and_t[p] = [r_p_of_t(p, t, h_int, k_trunc=k_trunc) for t in range(prob_levels)]
    # compute product over primes for each H
    prob_ge = {}
    for H in range(1, prob_levels+1):
        t = H-1
        prod = 1.0
        for p in selected_primes:
            prod *= r_by_p_and_t[p][t]
        prob_ge[H] = 1.0 - prod
    # Pr(h=k) = Pr(h>=k) - Pr(h>=k+1)
    probs_eq = {}
    for k in range(1, prob_levels+1):
        probs_eq[k] = prob_ge.get(k, 0.0) - prob_ge.get(k+1, 0.0)
    return prob_ge, probs_eq

if __name__ == "__main__":
    prob_ge, probs_eq = compute_probabilities(prime_cutoff=20000, K_max=200, k_trunc=60, prob_levels=6)
    print("Pr(h>=k):")
    for k in sorted(prob_ge):
        print(k, prob_ge[k])
    print("\nPr(h=k):")
    for k in sorted(probs_eq):
        print(k, probs_eq[k])
