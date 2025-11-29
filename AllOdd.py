# Fix overflow by computing sums with fractions or using pow with floats safely.
K = 80
Pmax = 20000

import math
from sympy import primerange, factorint

r = {1: 1}
def compute_r(n):
    if n in r:
        return r[n]
    fac = factorint(n)
    if 2 in fac:
        r[n] = 0
        return 0
    for exp in fac.values():
        if compute_r(exp) == 0:
            r[n] = 0
            return 0
    r[n] = 1
    return 1

for k in range(2, K+1):
    compute_r(k)

r_list = [r.get(k,0) for k in range(K+1)]

primes = list(primerange(2, Pmax+1))

def prime_factor_term(p):
    # compute sum with floats but using pow to avoid huge intermediate ints
    s = 0.0
    for k in range(1, K+1):
        if r_list[k]:
            s += p**(-(k+1))
    A = (1 - 1.0/p) + (p-1) * s
    return A

product = 1.0
for p in primes:
    product *= prime_factor_term(p)

# tail bound as before but compute safely
primes_tail = list(primerange(Pmax+1, Pmax*3 + 1000))

tail_bound_factor = 1.0
for p in primes_tail:
    denom = (1 - 1.0/p)
    add = p**(-(K+1))
    tail_bound_factor *= (1 + add/denom)

product, len(primes), tail_bound_factor
