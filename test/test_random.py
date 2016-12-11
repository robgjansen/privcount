#!/usr/bin/python

# a simple range and bin test for privcount's random constructions
# this test can not prove that these constructs produce uniformly random
# results, it can only reveal obvious range or partition issues

# this test will exit successfully, unless the counters are more than
# MAX_DIVERGENCE from the full range or equal bin counts

from random import SystemRandom

from privcount.counter import sample, sample_randint, derive_blinding_factor, counter_modulus

# Allow this much divergence from the full range and equal bin counts
MAX_DIVERGENCE = 0.02
MAX_DIVERGENCE_PERCENT = MAX_DIVERGENCE * 100.0

# some of the privcount random functions expect additional arguments
POSITIVE = True

# the hard-coded modulus value
PRIV_COUNTER_MODULUS = counter_modulus()

# the number of random values used in each trial
N_TRIALS = 100000

# the number of partitions used to check that values are uniformly distributed
BIN_COUNT = 2

# Each of these value functions should return a uniformly distributed
# pseudo-random value in [0, modulus)

def random_value(modulus):
    '''
    call python's random.randrange(modulus)
    '''
    # random.randrange() takes one argument: a maximum value
    # and returns a random value in [0, modulus)
    # it is *NOT* uniformy distributed in python versions < 3.2
    # but this test is not sophisticated enough to pick that up
    # https://docs.python.org/3.5/library/random.html#random.randrange
    return SystemRandom().randrange(modulus)

def sample_value(modulus):
    '''
    call privcount.util.sample with modulus
    '''
    # sample takes a modulus value, and returns a random value in [0, modulus)
    return sample(modulus)

def sample_randint_value(modulus):
    '''
    call privcount.util.sample_randint with 0 and modulus - 1
    '''
    # sample_randint takes values a and b, and returns a random value in [a, b]
    return sample_randint(0, modulus - 1)

def blinding_value(modulus):
    '''
    call privcount.util.blinding_value with modulus and POSITIVE
    '''
    # derive_blinding_factor takes a blinding factor, a modulus value, and a
    # boolean indicating whether to blind or unblind,
    # and returns a random value in [0, modulus) for blinding,
    # and (0, modulus] for unblinding
    # None means "generate a blinding factor"
    value = derive_blinding_factor(None, modulus, POSITIVE)
    if not POSITIVE:
        # adjust (0, modulus] to [0, modulus), ignoring modulus
        value -= 1
    return value

def range_trial(result_count, func, modulus):
    '''
    Observe the range of func(modulus) with random keys for result_count trials
    Return (min_value, max_value)
    '''
    min_value = None
    max_value = None
    count = 0
    while count < result_count:
        value = func(modulus)
        if min_value is None or value < min_value:
            min_value = value
        if max_value is None or value > max_value:
            max_value = value
        count += 1
    return (min_value, max_value)

def bin(value, modulus, n_bins):
    '''
    Bin a value in [0, modulus) into one of n_bins
    returns a bin number in [0, n_bins) corresponding to value, or None if
    value does not fit in any bin (this can happen if n_bins does not divide
    evenly into modulus)
    '''
    # find values that don't fit evenly in any bin
    residual = modulus % n_bins
    bin_width = (modulus - residual) // n_bins
    # make sure we didn't lose any values
    assert bin_width * n_bins + residual == modulus
    # make sure at least some values go in each bin
    assert bin_width > 0
    if value >= (modulus - residual):
        return None
    return value // bin_width

def func_bin(func, modulus, n_bins):
    '''
    Bin the value of func(modulus) into one of n_bins, returning the bin number
    returns the bin corresponding to func's output
    '''
    value = func(modulus)
    assert value >= 0L
    assert value < modulus
    bin_number = bin(value, modulus, n_bins)
    if bin_number is None:
        return None
    assert bin_number >= 0
    assert bin_number < n_bins
    return bin_number

def bin_trial(result_count, func, modulus, n_bins):
    '''
    Do result_count trials of func(modulus), and bin each result
    returns a list of bin counts
    '''
    bins = list(0 for i in range(n_bins))
    count = 0
    while count < result_count:
        bin_number = func_bin(func, modulus, n_bins)
        if bin_number is not None:
            bins[bin_number] += 1
            count += 1
    return bins

def format_difference(actual, expected, total):
    '''
    Calculate the difference between actual and expected
    Calculate the percentage of total that difference represents
    return a formatted string
    '''
    difference = actual - expected
    percentage = round(difference * 100.0 / total, 1)
    # with 100,000 trials, we shouldn't get any differences larger than ~2%
    assert percentage < MAX_DIVERGENCE_PERCENT
    return "{} - {} = {} ({} %)".format(actual, expected, difference,
                                        percentage)

def format_difference_list(actual_list, expected, total):
    '''
    Calculate the difference_list between actual and expected_list
    Calculate the percentage_list of total that difference_list represents
    return a formatted string
    '''
    difference_list = [actual - expected for actual in actual_list]
    percentage_list = [round(difference * 100.0 / total, 1)
                       for difference in difference_list]
    # with 100,000 trials, we shouldn't get any differences larger than ~2%
    for p in percentage_list:
        assert p < MAX_DIVERGENCE_PERCENT
    return "{} - {} = {} ({} %)".format(actual_list, expected, difference_list,
                                 percentage_list)

def run_trial(result_count, func, modulus, n_bins):
    '''
    Run a randomness trial on func and print the results
    '''
    (obs_min, obs_max) = range_trial(result_count, func, modulus)
    bin_list = bin_trial(result_count, func, modulus, n_bins)
    print "Actual - Expected = Difference (% Difference of modulus)"
    print "Min: {}".format(format_difference(obs_min, 0, modulus))
    print "modulus: {}".format(format_difference(obs_max, modulus, modulus))
    expected_bin_count = result_count / n_bins
    print "Bin: {}".format(format_difference_list(bin_list,
                                                  expected_bin_count,
                                                  expected_bin_count))

print "random.randrange:"
run_trial(N_TRIALS, random_value, PRIV_COUNTER_MODULUS, BIN_COUNT)
print ""

print "privcount.sample:"
run_trial(N_TRIALS, sample_value, PRIV_COUNTER_MODULUS, BIN_COUNT)
print ""

print "privcount.sample_randint:"
run_trial(N_TRIALS, sample_randint_value, PRIV_COUNTER_MODULUS, BIN_COUNT)
print ""

print "privcount.derive_blinding_factor:"
run_trial(N_TRIALS, blinding_value, PRIV_COUNTER_MODULUS, BIN_COUNT)
