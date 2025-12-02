# Copy & run in your own Python environment (requires only standard library + numpy)
import numpy as np
import random
from collections import defaultdict
import math

# --- Toy Kubilius-like model to compute p_ge_k ---
# This is a simple approximate model: assume for each prime p and each level
# the exponent contributes a Bernoulli(p^{-r}) style event. Replace with your exact pmf.

primes = [2,3,5,7,11,13,17,19,23,29,31,37,41]  # extend as needed
max_exp = 6  # consider prime^r with r<=max_exp
def prob_prime_divides_exponent(p, r):
    # toy approximation: probability that exponent has factor p^r approx p**(-r)
    return p**(-r)

# Model: generate offspring pmf for a node by sampling prime divisors of exponent
def single_node_offspring_prob(p):
    # approximate distribution for number of distinct prime divisors of exponent at p
    # we treat each q (prime) as independent Bernoulli with small prob sum_{r>=1} q^{-r}/r approx q^-1/(1-1/q)
    # but to keep it simple: use Poisson approximation with mean mu_p
    mu = 0.0
    for q in primes:
        if q==p: continue
        # contribution to expected number of distinct prime divisors (toy)
        mu += sum(prob_prime_divides_exponent(q, r) for r in range(1, max_exp+1))
    # convert to Poisson pmf
    lam = mu
    # return Poisson probabilities up to, say, 10 children
    pmf = [math.exp(-lam) * lam**k / math.factorial(k) for k in range(0, 12)]
    return pmf

# Rough survival probability after k generations in a Galton-Watson with mean m:
def approx_survival_probability(mean_offspring, k):
    # For a subcritical Galton-Watson (m<1), survival prob decays roughly like m^k.
    return mean_offspring**k if mean_offspring>0 else 0.0

# compute toy p_ge_k and p_eq_k
means = []
for p in primes:
    pmf = single_node_offspring_prob(p)
    m = sum(i*pmf[i] for i in range(len(pmf)))
    means.append(m)
mean_offspring = np.mean(means)  # toy aggregate
p_ge = {}
for k in range(1,7):
    p_ge[k] = approx_survival_probability(mean_offspring, k)
p_eq = {k: p_ge[k] - p_ge.get(k+1,0) for k in range(1,7)}

print("toy mean_offspring:", mean_offspring)
print("toy p_ge (prob height >= k):", p_ge)
print("toy p_eq (prob height == k):", p_eq)

# --- Empirical test: compute heights for integers up to X (toy height function) ---
def toy_height(n):
    # crude toy: height is number of times you can iterate "take number of distinct prime divisors"
    # until you get 1. This is NOT the real tower factorization height; replace with your exact routine.
    def distinct_prime_divisors(x):
        cnt = 0
        t = x
        for p in primes:
            if p*p>t: break
            if t%p==0:
                cnt+=1
                while t%p==0: t//=p
        if t>1 and t in primes:
            cnt+=1
        return cnt if cnt>0 else 1
    h = 0
    t = n
    while t>1 and h<10:
        t = distinct_prime_divisors(t)
        h += 1
    return h

# simulate up to X
X = 100000
heights = [toy_height(n) for n in range(1, X+1)]

def gaps_for_height(k):
    positions = [i+1 for i,h in enumerate(heights) if h==k]
    if not positions: return []
    gaps = [positions[0]-1] + [positions[i]-positions[i-1] for i in range(1,len(positions))]
    return gaps

k = 2
gaps = gaps_for_height(k)
print("empirical mean gap for k=",k, ":", np.mean(gaps) if gaps else None)
# histogram vs geometric
if gaps:
    vals, counts = np.unique(gaps, return_counts=True)
    freqs = counts / counts.sum()
    print("small gap freq (empirical):", list(zip(vals[:10], freqs[:10])))
    # geometric approx parameter:
    p_hat = 1.0/np.mean(gaps)
    print("geometric param p_hat:", p_hat)
