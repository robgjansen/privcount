#!/usr/bin/python

# this test will fail if any counter inconsistencies are detected
# 

import sys

from privcount.counter import SecureCounters, adjust_count_signed, counter_modulus, add_counter_limits_to_config, get_events_for_known_counters
from math import sqrt
from random import SystemRandom

import logging
# DEBUG logs every check: use it on failure
# INFO logs each major counter size increase
logging.basicConfig(level=logging.INFO)
logging.root.name = ''

# When testing counter modulus values, use this range
modulus_min = 1L
# test at least 2**64, or 2**5 times our current limit
modulus_max = max(2L**64L, counter_modulus() * 2L**5L)

# A simple set of byte counters
counters = {
  'SanityCheck': {
    'bins':
    [
      [0.0, float('inf')],
    ],
    'sigma': 0.0
  },
  'Bytes': {
    'bins':
    [
      [0.0, 512.0],
      [512.0, 1024.0],
      [1024.0, 2048.0],
      [2048.0, 4096.0],
      [4096.0, float('inf')],
    ],
    'sigma': 0.0
  }
}

def check_adjust_count_signed(modulus):
    '''
    check that adjust_count_signed works as expected for modulus
    '''
    # for any modulus, returns { (modulus + 1)//2 - modulus, ... , 0, ... , (modulus + 1)//2 - 1 }
    assert adjust_count_signed(0, modulus) == 0
    # we assume here that modulus is at least large enough to have -1, 0, 1
    if modulus < 3L:
        return
    assert adjust_count_signed(1L, modulus) == 1
    assert adjust_count_signed((modulus + 1L)//2L - 1L, modulus) == (modulus + 1L)//2L - 1L
    assert adjust_count_signed((modulus + 1L)//2L, modulus) == (modulus + 1L)//2L - modulus
    assert adjust_count_signed(modulus - 1L, modulus) == -1L

def try_adjust_count_signed(modulus):
    '''
    check that adjust_count_signed works as expected for modulus,
    and a randomly chosen number between modulus_min and modulus
    '''
    # randrange is not uniformly distributed in python versions < 3.2
    modulus_random = SystemRandom().randrange(modulus_min, modulus)
    check_adjust_count_signed(modulus_random)
    check_adjust_count_signed(modulus)

def check_blinding_values(secure_counters, modulus):
    '''
    check that each blinding value is unique
    if not, there is a coding error that affects the security of the system
    '''
    blinding_value_list = []
    for key in secure_counters.counters:
        for item in secure_counters.counters[key]['bins']:
            blinding_value_list.append(item[2])
    # are all the blinding values unique?
    # only check if the number of items is very small compared with modulus
    # RAM bit error rates are up to 10^-13 per second
    # http://www.cs.toronto.edu/~bianca/papers/sigmetrics09.pdf
    # So this check should fail by chance less often than a hardware error
    # Using the approximation from
    # https://en.wikipedia.org/wiki/Birthday_problem#Square_approximation
    # n = sqrt(2m * p(n)) where m is modulus and p(n) is 10^-13
    max_blinding_value_count = sqrt(2.0 * (modulus / (10L**13L)))
    blinding_value_count = len(blinding_value_list)
    unique_blinding_value_count = len(set(blinding_value_list))
    logging.debug("modulus: {} max_blinding: {} actual blinding: {}".format(
                      modulus, max_blinding_value_count, blinding_value_count))
    if blinding_value_count < max_blinding_value_count:
        assert blinding_value_count == unique_blinding_value_count
    else:
       logging.debug("skip blinding collision check: collisions too likely")

def create_counters(counters, modulus):
    '''
    create the counters for a data collector, who will generate the shares and
    noise
    uses modulus to generate the appropriate blinding factors
    returns a tuple containing a list of DCs and a list of SKs
    '''
    sc_dc = SecureCounters(counters, modulus)
    sc_dc.generate_blinding_shares(['sk1', 'sk2'])
    sc_dc.generate_noise(1.0)
    check_blinding_values(sc_dc, modulus)
    # get the shares used to init the secure counters on the share keepers
    shares = sc_dc.detach_blinding_shares()

    # create share keeper versions of the counters
    sc_sk1 = SecureCounters(counters, modulus)
    sc_sk1.import_blinding_share(shares['sk1'])
    check_blinding_values(sc_sk1, modulus)
    sc_sk2 = SecureCounters(counters, modulus)
    sc_sk2.import_blinding_share(shares['sk2'])
    check_blinding_values(sc_sk2, modulus)
    return ([sc_dc], [sc_sk1, sc_sk2])

def increment_counters(dc_list, N, multi_bin=True):
    '''
    run a set of increments at the dc N times, using the two-argument form of
    increment() to increment by 1 each time
    each bin has 0, 1, or 2 values added per increment when multi_bin is True,
    and 0 or 1 values added per increment when multi_bin is False
    Returns long(N)
    '''
    sc_dc = dc_list[0]
    # xrange only accepts python ints, which is ok, because it's impossible to
    # increment more than 2**31 times in any reasonable test duration
    assert N <= sys.maxint
    for _ in xrange(int(N)):
        # bin[0]
        sc_dc.increment('Bytes', 0.0)
        if multi_bin:
            sc_dc.increment('Bytes', 511.0)
        #bin[1]
        sc_dc.increment('Bytes', 600.0)
        #bin[2]
        if multi_bin:
            sc_dc.increment('Bytes', 1024.0)
        sc_dc.increment('Bytes', 2047.0)
        #bin[3]
        pass
        #bin[4]
        sc_dc.increment('Bytes', 4096.0)
        if multi_bin:
            sc_dc.increment('Bytes', 10000.0)
    return long(N)

def increment_counters_num(dc_list, N, X=1L, multi_bin=True):
    '''
    run a set of increments at the dc N times, incrementing by X each time
    each bin has 0, 1, or 2 values added per increment when multi_bin is True,
    and 0 or 1 values added per increment when multi_bin is False
    Returns long(N) * long(X)
    '''
    sc_dc = dc_list[0]
    # xrange only accepts python ints, which is ok, because it's impossible to
    # increment more than 2**31 times in any reasonable test duration
    assert N <= sys.maxint
    for _ in xrange(int(N)):
        # bin[0]
        # test that increment handles signed numbers, doubles & ints correctly
        # we can't rely on X being able to fit in a double or an int
        # so just increment by one, then subtract one
        sc_dc.increment('Bytes', 0.0, 1.0)
        sc_dc.increment('Bytes', 0.0, -1)
        sc_dc.increment('Bytes', 0.0, long(X))
        if multi_bin:
            sc_dc.increment('Bytes', 511.0, long(X))
        #bin[1]
        sc_dc.increment('Bytes', 600.0, long(X))
        #bin[2]
        if multi_bin:
            sc_dc.increment('Bytes', 1024.0, long(X))
        sc_dc.increment('Bytes', 2047.0, long(X))
        #bin[3]
        pass
        #bin[4]
        sc_dc.increment('Bytes', 4096.0, long(X))
        if multi_bin:
            sc_dc.increment('Bytes', 10000.0, long(X))
    return long(N)*long(X)

def sum_counters(counters, modulus, dc_list, sk_list):
    '''
    Sums the counters in dc_list and sk_list, with maximum count modulus
    Returns a tallies object populated with the resulting counts
    '''
    # get all of the counts, send for tallying
    counts_dc_list = [sc_dc.detach_counts() for sc_dc in dc_list]
    counts_sk_list = [sc_sk.detach_counts() for sc_sk in sk_list]

    # tally them up
    sc_ts = SecureCounters(counters, modulus)
    counts_list = counts_dc_list + counts_sk_list
    is_tally_success = sc_ts.tally_counters(counts_list)
    assert is_tally_success
    return sc_ts.detach_counts()

def check_counters(tallies, N, multi_bin=True):
    '''
    Checks that the tallies are the expected values, based on the number of
    repetitions N, and whether we're in multi_bin mode
    '''
    logging.debug("amount: {}".format(N))
    logging.debug("tallies: {}".format(tallies))
    # these assertions may also fail if the counter values overflow modulus
    if multi_bin:
        assert tallies['Bytes']['bins'][0][2] == 2*N
    else:
        assert tallies['Bytes']['bins'][0][2] == 1*N
    assert tallies['Bytes']['bins'][1][2] == 1*N
    if multi_bin:
        assert tallies['Bytes']['bins'][2][2] == 2*N
    else:
        assert tallies['Bytes']['bins'][2][2] == 1*N
    assert tallies['Bytes']['bins'][3][2] == 0*N
    if multi_bin:
        assert tallies['Bytes']['bins'][4][2] == 2*N
    else:
        assert tallies['Bytes']['bins'][4][2] == 1*N
    assert tallies['SanityCheck']['bins'][0][2] == 0
    logging.debug("all counts are correct!")

def run_counters(counters, modulus, N, X=None, multi_bin=True):
    '''
    Validate that a counter run with counters, modulus, N, X, and multi_bin works,
    and produces consistent results
    If X is None, use the 2-argument form of increment, otherwise, use the
    3-argument form
    '''
    logging.debug("modulus: {} N: {} X: {} multi_bin: {}".format(
                      modulus, N,
                      X if X is not None else "None",
                      multi_bin))
    (dc_list, sk_list) = create_counters(counters, modulus)
    if X is None:
        # use the 2-argument form
        amount = increment_counters(dc_list, N, multi_bin)
        assert amount == N
    else:
        # use the 3-argument form
        amount = increment_counters_num(dc_list, N, X, multi_bin)
        assert amount == N*X
    tallies = sum_counters(counters, modulus, dc_list, sk_list)
    check_counters(tallies, amount, multi_bin)

def try_counters(counters, modulus, N, X=None, multi_bin=True):
    '''
    Validate that a counter run with counters, modulus, N, X, and multi_bin works,
    and produces consistent results
    Also try a randomly selected number modulus_random between modulus_min and modulus,
    and a randomly selected number N_random between 0 and min(q_random, N)
    and a randomly selected number X_random between 0 and min(q_random, X)
    If X is None, use the 2-argument form of increment, otherwise, use the
    3-argument form
    '''
    # randrange is not uniformly distributed in python versions < 3.2
    modulus_random = SystemRandom().randrange(modulus_min, modulus)
    N_random = SystemRandom().randrange(0, min(modulus_random, N))
    X_random = None
    if X is not None:
        X_random = SystemRandom().randrange(0, min(modulus_random, X))
    run_counters(counters, modulus_random, N_random, X_random, multi_bin)
    run_counters(counters, modulus, N, X, multi_bin)

# Check the counter table is valid, and perform internal checks
assert len(get_events_for_known_counters()) > 0

# Check that unsigned to signed conversion works with odd and even modulus
logging.info("Unsigned to signed counter conversion, modulus = 3:")
# for odd  modulus, returns { -modulus//2, ... , 0, ... , modulus//2 }
assert adjust_count_signed(0L, 3L) == 0L
assert adjust_count_signed(1L, 3L) == 1L
assert adjust_count_signed(2L, 3L) == -1L
logging.info("Success!")

logging.info("Unsigned to signed counter conversion, modulus = 4:")
# for even modulus, returns { -modulus//2, ... , 0, ... , modulus//2 - 1 }
assert adjust_count_signed(0L, 4L) == 0L
assert adjust_count_signed(1L, 4L) == 1L
assert adjust_count_signed(2L, 4L) == -2L
assert adjust_count_signed(3L, 4L) == -1L
logging.info("Success!")

logging.info("Unsigned to signed counter conversion, modulus = {}:".format(
                 counter_modulus()))
try_adjust_count_signed(counter_modulus())
logging.info("Success!")

logging.info("Unsigned to signed counter conversion, modulus = {}:".format(
                 counter_modulus() + 1L))
try_adjust_count_signed(counter_modulus() + 1L)
logging.info("Success!")

logging.info("Unsigned to signed counter conversion, modulus = {}:".format(
                 counter_modulus() - 1L))
try_adjust_count_signed(counter_modulus() - 1L)
logging.info("Success!")


# Check that secure counters increment correctly for small values of N
# using the default increment of 1
logging.info("Multiple increments, 2-argument form of increment:")
N = 500L
try_counters(counters, counter_modulus(), N)

# Check that secure counters increment correctly for a single increment
# using a small value of num_increment
logging.info("Single increment, 3-argument form of increment:")
N = 1L
X = 500L
try_counters(counters, counter_modulus(), N, X)

# And multiple increments using the 3-argument form
logging.info("Multiple increments, 3-argument form of increment, explicit +1:")
N = 500L
X = 1L
try_counters(counters, counter_modulus(), N, X)

logging.info("Multiple increments, 3-argument form of increment, explicit +2:")
N = 250L
X = 2L
try_counters(counters, counter_modulus(), N, X)

logging.info(
    "Multiple increments, 2-argument form of increment, multi_bin=False:")
N = 20L
try_counters(counters, counter_modulus(), N, multi_bin=False)

logging.info(
    "Multiple increments, 3-argument form of increment, multi_bin=False:")
N = 20L
X = 1L
try_counters(counters, counter_modulus(), N, X, multi_bin=False)


logging.info("Increasing increments, designed to trigger an overflow:")
N = 1L

a = 1L
X = 2L**a
# we interpret values >= modulus/2 as negative, so there's no point testing them
# (privcount requires that the total counts are much less than modulus/2)
while X < counter_modulus()//2L:
    logging.info("Trying count 2**{} = {} < modulus/2 ({})".format(
                     a, X, counter_modulus()//2))
    try_counters(counters, counter_modulus(), N, X, multi_bin=False)
    # This should terminate in at most ~log2(modulus) steps
    a += 1L
    X = 2L**a

# Now try modulus/2-1 explicitly
N = 1L
X = counter_modulus()//2L - 1L
logging.info("Trying count = modulus/2 - 1 = {}".format(X))
try_counters(counters, counter_modulus(), N, X, multi_bin=False)
logging.info("Reached count of modulus/2 - 1 = {} without overflowing".format(
                 X))


logging.info("Increasing modulus, designed to trigger floating-point inaccuracies:")
N = 1L

b = 1L
q_try = 2L**b

assert modulus_max >= modulus_min
while q_try <= modulus_max:
    # Skip the sampling if modulus is too low
    if q_try >= modulus_min:
        a = 1L
        X = 2L**a
        logging.info("Trying modulus = 2**{} = {}".format(b, q_try))
        logging.info(
            "Unsigned to signed counter conversion, modulus = {}:".format(
                q_try))
        try_adjust_count_signed(q_try)
        logging.info("Success!")
        # we interpret values >= modulus/2 as negative
        # So make sure that modulus is larger than 2*(N*X + 1)
        while X < q_try//2L:
            logging.debug("Trying count 2**{} = {} < modulus/2 ({})".format(
                              a, X, q_try//2))
            try_counters(counters, q_try, N, X, multi_bin=False)
            # This inner loop should terminate in at most ~log2(modulus) steps
            a += 1L
            X = 2L**a
    # This outer loop should terminate in at most ~log2(modulus_max) steps
    b += 1L
    q_try = 2L**b

# Now try modulus = modulus_max explicitly
N = 1L
q_try = modulus_max
assert modulus_max >= modulus_min
a = 1L
X = 2L**a
logging.info("Trying modulus = modulus_max = {}".format(q_try))
logging.info("Unsigned to signed counter conversion, modulus = {}:".format(
                 q_try))
try_adjust_count_signed(q_try)
logging.info("Success!")
# we interpret values >= modulus/2 as negative
# So make sure that modulus is larger than 2*(N*X + 1)
while X < q_try//2L:
    logging.debug("Trying count 2**{} = {} < modulus/2 ({})".format(
                      a, X, q_try//2))
    try_counters(counters, q_try, N, X, multi_bin=False)
    # This should terminate in at most ~log2(modulus) steps
    a += 1L
    X = 2L**a

logging.info(
    "Reached modulus = modulus_max = {} without overflow or inaccuracy".format(
        q_try))

logging.info("Hard-coded counter values: {}".format(
                 add_counter_limits_to_config({})))
