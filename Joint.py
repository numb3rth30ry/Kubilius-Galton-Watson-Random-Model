# Compute probability that all primes in S appear in TF(n)
# Hybrid Kubilius fixed-point method; parameters pre-filled.

import math
from functools import lru_cache
from itertools import combinations

# -----------------------------
# helpers (same as before)
# -----------------------------
def primes_upto(N):
    sieve = bytearray(b'\x01')*(N+1)
    sieve[0:2]=b'\x00\x00'
    for p in range(2, int(N**0.5)+1):
        if sieve[p]:
            start = p*p
            sieve[start:N+1:p] = b'\x00' * (((N - start)//p) + 1)
    return [i for i,isprime in enumerate(sieve) if isprime]

def Omega(n):
    if n <= 1: return 0
    cnt = 0
    temp = n
    p = 2
    while p*p <= temp:
        while temp % p == 0:
            cnt += 1; temp //= p
        p += 1 if p==2 else 2
    if temp > 1: cnt += 1
    return cnt

@lru_cache(None)
def tf_contains_q(m, q):
    # True iff q appears anywhere in TF(m)
    if m <= 1: return False
    temp = m; p = 2
    while p*p <= temp:
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                e += 1; temp //= p
            if p == q: return True
            if tf_contains_q(e, q): return True
        p += 1 if p==2 else 2
    if temp > 1:
        if temp == q: return True
    return False

# Precompute g_T(e) = 1 iff none of primes in T appear in TF(e)
def precompute_g(E_MAX, T):
    g = [None]*(E_MAX+1)
    for e in range(E_MAX+1):
        absent = True
        for q in T:
            if tf_contains_q(e, q):
                absent = False
                break
        g[e] = 1 if absent else 0
    return g

# Fixed-point solver to compute beta_T = probability none of T appear
def compute_beta_T(T, primes, E_MAX, tol=1e-11, maxiter=200):
    Tset = set(T)
    g = precompute_g(E_MAX, tuple(T))
    omega_cache = {e: Omega(e) for e in range(E_MAX+1)}

    def inner_sum(p, alpha):
        s = 0.0
        one_minus = 1.0 - 1.0/p
        # e = 0 term (w(0)=1)
        s += one_minus * 1.0
        # if p is one of the banned primes, only e=0 contributes
        if p in Tset:
            return s
        p_pow = p
        for e in range(1, E_MAX+1):
            term = one_minus * (1.0 / p_pow) * g[e]
            s += term
            p_pow *= p
        # tail: use alpha^{Omega(e)} approximation
        e_tail_max = 200
        for e in range(E_MAX+1, e_tail_max+1):
            om = omega_cache.get(e)
            if om is None:
                om = Omega(e); omega_cache[e] = om
            term = one_minus * (1.0 / p_pow) * (alpha ** om)
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
            if val <= 0: val = 1e-300
            log_prod += math.log(val)
        alpha_new = math.exp(log_prod)
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new
    return alpha  # this is beta_T

# -----------------------------
# Parameters (same stable choices)
# -----------------------------
P_MAX = 10000          # primes up to 10k
primes = primes_upto(P_MAX)
E_MAX = 50             # compute exact g(e) for e <= 50
tol = 1e-11
maxiter = 200

# -----------------------------
# Choose S here
# -----------------------------
S = [2, 3]   # example set; change as needed

# -----------------------------
# Compute beta_T for each subset T of S
# -----------------------------
subsets = [()]
for k in range(1, len(S)+1):
    for comb in combinations(S, k):
        subsets.append(tuple(comb))

betas = {}
for T in subsets:
    beta = compute_beta_T(list(T), primes, E_MAX, tol=tol, maxiter=maxiter)
    betas[T] = beta
    print(f"beta_{T} = {beta:.12f}")

# -----------------------------
# Inclusion-exclusion: prob all primes in S appear
# -----------------------------
prob_all = 0.0
for T in subsets:
    sign = (-1)**(len(T))
    prob_all += sign * betas[T]
print("\nProbability every prime in S appears:")
print(f"S = {S};  Pr(all appear) = {prob_all:.12f}")
