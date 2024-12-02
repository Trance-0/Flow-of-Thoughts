import random
import time
import math
import numpy as np

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def modinv(a, m):
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

def rsa_pseudo_random_generator(seed, n_bits):
    p = 61
    q = 53
    n = p * q
    e = 17
    d = modinv(e, (p-1)*(q-1))
    x = seed
    random_bits = []
    for _ in range(n_bits):
        x = pow(x, e, n)
        random_bits.append(x % 2)
    return random_bits

def measure_time_rsa(seed, n_bits,verbose=False):
    start_time = time.time()
    res=rsa_pseudo_random_generator(seed, n_bits)
    end_time = time.time()
    if verbose:
        print(f"RSA Pseudo Random Generator {res}")
    return end_time - start_time

def measure_time_python(n_bits,verbose=False):
    start_time = time.time()
    res=[random.getrandbits(1) for _ in range(n_bits)]
    end_time = time.time()
    if verbose:
        print(f"Python Random Generator {res}")
    return end_time - start_time

def measure_time_np(n_bits, verbose=False):
    start_time = time.time()
    res = np.random.uniform(-1, 1, n_bits)
    end_time = time.time()
    if verbose:
        print(f"NumPy Random Generator {res}")
    return end_time - start_time

seed = 42
n_bits = 10000000

rsa_time = measure_time_rsa(seed, n_bits,verbose=True)
python_time = measure_time_python(n_bits,verbose=True)
np_time = measure_time_np(n_bits,verbose=True)

print(f"RSA Pseudo Random Generator Time: {rsa_time} seconds")
print(f"Python Random Generator Time: {python_time} seconds")
print(f"NP random Generator Time: {np_time} seconds")