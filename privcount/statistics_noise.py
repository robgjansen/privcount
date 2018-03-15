# See LICENSE for licensing information

import logging
import math
import yaml

from privcount.counter import DEFAULT_SIGMA_TOLERANCE, DEFAULT_EPSILON_TOLERANCE, DEFAULT_SIGMA_RATIO_TOLERANCE, DEFAULT_DUMMY_COUNTER_NAME, is_circuit_sample_counter
from privcount.log import summarise_list

def satisfies_dp(sensitivity, epsilon, delta, std):
    '''
    Return True if (epsilon, delta)-differential privacy is satisfied.
    '''
    # find lowest value at which epsilon differential-privacy is satisfied
    lower_x = -(float(epsilon) * (std**2.0) / sensitivity) + sensitivity/2.0
    # determine lower tail probability of normal distribution w/ mean of zero
    lower_tail_prob = lower_tail_prob = (1.0 + math.erf(lower_x / std / math.sqrt(2.0))) / 2.0
    # explicitly return Boolean value to avoid returning numpy type
    if (lower_tail_prob <= delta):
        return True
    else:
        return False

def interval_boolean_binary_search(fn, lower_bound, upper_bound, tol,
                                   return_true=True):
    '''
    Searches interval (lower_bound, upper_bound) for x such that fn(x) is
    True. Assumes fn is monotonic: if x<y and fn(x)=True, then fn(y) is True.
    If return_true is True, returns smallest value x such that fn(x)=True,
    within tolerance tol. Else returns largest value such that fn(y)=False,
    within tolerance tol.
    Unlike similar functions, tol does not have a default - insted, use the
    default tolerance for the specific variable being searched.
    '''
    if (upper_bound < lower_bound):
        raise ValueError('Invalid binary-search interval: [{}, {}]'.format(\
            lower_bound, upper_bound))

    if (fn(lower_bound) is True):
        if return_true is True:
            return lower_bound
        else:
            raise ValueError('Can\'t return x False, fn(lower_bound)=True.')
    if (fn(upper_bound) is False):
        if return_true is True:
            raise ValueError('Can\'t return x True, fn(upper_bound)=False.')
        else:
            return upper_bound

    # iteratively search for satisfying input x
    while True:
        diff_before = upper_bound - lower_bound
#        print('Interval search lower: {}, upper: {}, diff: {}, tol: {}'.format(lower_bound, upper_bound, diff_before, tol))
        if (diff_before < tol):
            if return_true is True:
                return upper_bound
            else:
                return lower_bound
        if '{}'.format(upper_bound) == '{}'.format(lower_bound):
            return upper_bound
        midpoint = upper_bound - (diff_before/2.0)
        midpoint_val = fn(midpoint)
#        print('Midpoint: {}, val: {}'.format(midpoint, midpoint_val))
        if (midpoint_val is False):
            lower_bound = midpoint
        else:
            upper_bound = midpoint
        diff_after = upper_bound - lower_bound
        if diff_before == diff_after and '{}'.format(upper_bound) == '{}'.format(lower_bound):
            return upper_bound


def get_differentially_private_std(sensitivity, epsilon, delta,
                                   tol=DEFAULT_SIGMA_TOLERANCE):
    '''
    Determine smallest standard deviation for a normal distribution such that
    the probability of a value violating epsilon-differential privacy is at
    most delta.
    '''

    # std upper bound determined by improving result in literature,
    # Hardt and Roth, "Beating Randomized Response on Incoherent Matrices"
    # Thm. 2.6 (and the Lemma in App. A) can be improved to provide the
    # following upper bound
    std_upper_bound = (float(sensitivity)/epsilon) * (4.0/3.0) *\
        (2 *  math.log(1.0/delta))**(0.5)
    std_lower_bound = tol # use small but non-zero value for std lower-bound
    if (satisfies_dp(sensitivity, epsilon, delta, std_lower_bound) is True):
        raise ValueError('Could not find lower bound for std interval.')

    std = interval_boolean_binary_search(\
        lambda x: satisfies_dp(sensitivity, epsilon, delta, x), std_lower_bound,
        std_upper_bound, tol, return_true=True)

    return std

def get_approximate_privacy_allocation(epsilon, delta, stats_parameters,
                                       sigma_tol=DEFAULT_SIGMA_TOLERANCE):
    '''
    Allocate privacy budget among statistics
    Allocate epsilon to equalize noise ratios assuming delta is shared
    equally. Find optimal sigma (within tolerance sigma_tol) given epsilon
    privacy allocation.
    '''
    # allocate epsilon
    epsilons = dict()
    init_constant = None
    init_param = None
    coefficient_sum = 1
    for param, (s, v) in stats_parameters.iteritems():
        # ignore dummy counters
        if v == 0.0:
            continue
        if init_constant is None:
            init_constant = float(s) / v
            init_param = param
            continue
        coefficient_sum += float(s) / v / init_constant
    epsilons[init_param] = float(epsilon)/coefficient_sum
    for param, (s, v) in stats_parameters.iteritems():
        if (param != init_param):
            # give dummy counters a sensible default value
            if v == 0.0:
                epsilons[param] = get_sanity_check_counter()['epsilon']
            else:
                epsilons[param] = (epsilons[init_param] * float(s) / v /
                                   init_constant)
    # determine sigmas to acheive desired epsilons and delta
    sigmas = dict()
    stat_delta = float(delta) / len(stats_parameters)
    for param, (s, v) in stats_parameters.iteritems():
        # give dummy counters a sensible default value
        if s == 0.0:
            sigmas[param] = get_sanity_check_counter()['sigma']
        else:
            sigma = get_differentially_private_std(s, epsilons[param],
                                                   stat_delta, tol=sigma_tol)
            sigmas[param] = sigma

    return (epsilons, sigmas)

def get_differentially_private_epsilon(sensitivity, sigma, delta,
                                       tol=DEFAULT_EPSILON_TOLERANCE):
    '''
    find epsilons for fixed delta
    '''
    # skip search for sanity check counters
    if sigma == 0.0:
        return 0.0
    epsilon_upper_bound = (float(sensitivity)/sigma) * (2.0 *  math.log(2.0/delta))**(0.5)
    epsilon_lower_bound = 0.0
    epsilon = interval_boolean_binary_search(\
        lambda x: satisfies_dp(sensitivity, x, delta, sigma), epsilon_lower_bound,
        epsilon_upper_bound, tol, return_true=True)

    return epsilon

def get_epsilon_consumed(stats_parameters, excess_noise_ratio, sigma_ratio,
                         delta, tol=DEFAULT_EPSILON_TOLERANCE):
    '''
    given sigma, determine total epsilon used
    '''
    stat_delta = float(delta) / len(stats_parameters)
    epsilons = dict()
    for param, (sensitivity, val) in stats_parameters.iteritems():
        sigma = get_sigma(excess_noise_ratio, sigma_ratio, val)
        epsilon = get_differentially_private_epsilon(sensitivity, sigma, stat_delta, tol=tol)
        epsilons[param] = epsilon

    return epsilons

def get_sigma(excess_noise_ratio, sigma_ratio, estimated_value):
    '''
    Calculate the (optimal) sigma from the excess noise ratio, (optimal)
    sigma ratio, and per-statistic expected value.
    Inverse of get_expected_noise_ratio.
    '''
    if excess_noise_ratio == 0.0:
        return 0.0
    else:
        # let negative excess_noise_ratio raise an exception
        return (float(sigma_ratio) * estimated_value /
                math.sqrt(excess_noise_ratio))

def get_expected_noise_ratio(excess_noise_ratio, sigma, estimated_value):
    '''
    Calculate the expected noise ratio from the excess noise ratio,
    and per-statistic sigma and expected value.
    Inverse of get_sigma.
    '''
    if estimated_value == 0.0:
        return 0.0
    else:
        # let negative excess_noise_ratio raise an exception
        return math.sqrt(excess_noise_ratio) * sigma / estimated_value

def get_opt_privacy_allocation(epsilon, delta, stats_parameters,
                               excess_noise_ratio,
                               sigma_tol=DEFAULT_SIGMA_TOLERANCE,
                               epsilon_tol=DEFAULT_EPSILON_TOLERANCE,
                               sigma_ratio_tol=DEFAULT_SIGMA_RATIO_TOLERANCE):
    '''
    search for sigma ratio (and resulting epsilon allocation) that just
    consumes epsilon budget
    '''
    # get allocation that is optimal for approximate sigmas to get sigma ratio bounds
    approx_epsilons, approx_sigmas = get_approximate_privacy_allocation(epsilon, delta,
        stats_parameters, sigma_tol=sigma_tol)
    # ratios of sigma to expected value
    min_sigma_ratio = None
    max_sigma_ratio = None
    for param, (sensitivity, val) in stats_parameters.iteritems():
        ratio = get_expected_noise_ratio(excess_noise_ratio,
                                         approx_sigmas[param],
                                         val)
        if (min_sigma_ratio is None) or (ratio < min_sigma_ratio):
            min_sigma_ratio = ratio
        if (max_sigma_ratio is None) or (ratio > max_sigma_ratio):
            max_sigma_ratio = ratio
    # get optimal sigma ratio
    opt_sigma_ratio = interval_boolean_binary_search(\
        lambda x: sum((get_epsilon_consumed(stats_parameters, excess_noise_ratio, x, delta,
            tol=epsilon_tol)).itervalues()) <= epsilon,
        min_sigma_ratio, max_sigma_ratio, sigma_ratio_tol, return_true=True)
    # compute epsilon allocation that achieves optimal sigma ratio
    opt_epsilons = get_epsilon_consumed(stats_parameters, excess_noise_ratio, opt_sigma_ratio,
        delta, tol=epsilon_tol)
    # turn opt sigma ratio into per-parameter sigmas
    opt_sigmas = dict()
    zero_sigmas = []
    low_sigmas = []
    for param, (sensitivity, val) in stats_parameters.iteritems():
        opt_sigma = get_sigma(excess_noise_ratio, opt_sigma_ratio, val)
        # Check if the sigma is too small
        if param != DEFAULT_DUMMY_COUNTER_NAME:
            if opt_sigma == 0.0:
                zero_sigmas.append(param)
            elif opt_sigma < DEFAULT_SIGMA_TOLERANCE:
                low_sigmas.append(param)
        opt_sigmas[param] = opt_sigma

    if len(zero_sigmas) > 0:
        logging.error("sigmas for {} are zero, this provides no differential privacy for these statistics"
                      .format(summarise_list(zero_sigmas)))

    if len(low_sigmas) > 0:
        logging.warning("sigmas for {} are less than the sigma tolerance {}, their calculated values may be inaccurate and may vary each time they are calculated"
                        .format(summarise_list(low_sigmas), sigma_tol))

    return (opt_epsilons, opt_sigmas, opt_sigma_ratio)

def get_sanity_check_counter():
    '''
    Provide a dictionary with the standard sanity check counter values.
    It is typically used like:
        counters['ZeroCount'] = get_sanity_check_counter()
    All of these values are unused, except for:
        bins:
            - [-inf, inf] (a long counter is appended by SecureCounters,
                           it should only ever have blinding values added)
        estimated_value: 0.0 (TODO: used for checking if stats have changed)
        sigma: 0.0 (used for adding noise, 0.0 means no noise is added)
    '''
    sanity_check = {}
    sanity_check['bins'] = []
    single_bin = [float('-inf'), float('inf')]
    sanity_check['bins'].append(single_bin)
    sanity_check['sensitivity'] = 0.0
    sanity_check['estimated_value'] = 0.0
    sanity_check['sigma'] = 0.0
    sanity_check['epsilon'] = 0.0
    sanity_check['expected_noise_ratio'] = 0.0
    return sanity_check

def get_noise_allocation(noise_parameters,
                         sanity_check=DEFAULT_DUMMY_COUNTER_NAME,
                         circuit_sample_rate=1.0):
    '''
    An adapter which wraps get_opt_privacy_allocation, extracting the
    parameters from the noise_parameters data structure, and updating
    noise_parameters with the calculated values.
    If sanity_check is not None, adds a sanity check counter to the result,
    with the counter name supplied in sanity_check, and values created using
    get_sanity_check_counter().
    Scales expected circuit counter values by circuit_sample_rate before
    allocating noise.
    Returns a data structure containing the results on success.
    Raises a ValueError on failure.
    The format of noise_parameters is:
    privacy:
        epsilon: float in
        delta: float in
        excess_noise_ratio: float in
        sigma_tolerance: float in optional default 1e-6
        epsilon_tolerance: float in optional default 1e-15
        sigma_ratio_tolerance: float in optional default 1e-6
        sigma_ratio: float out
    counters:
        'CounterName': multiple
            bins: optional, unused
                - [float, float, long optional] multiple, unused
            sensitivity: float in
            estimated_value: float in
            sigma: float out
            epsilon: float out
            expected_noise_ratio: float out
    The expected noise ratio should be identical for each counter, except for
    floating-point inaccuracies.
    '''
    assert circuit_sample_rate >= 0.0
    assert circuit_sample_rate <= 1.0
    # extract the top-level structures
    noise = noise_parameters['privacy']
    counters = noise_parameters['counters']
    excess_noise_ratio = noise['excess_noise_ratio']
    # rearrange the counter values, and produce the parameter-only structure
    stats_parameters = {}
    zero_sigmas = []
    for stat in counters:
        sensitivity = counters[stat]['sensitivity']
        estimated_value = counters[stat]['estimated_value']
        if is_circuit_sample_counter(stat):
            estimated_value *= circuit_sample_rate
        if sensitivity == 0 and stat != DEFAULT_DUMMY_COUNTER_NAME:
            zero_sigmas.append(stat)
        statistics = (sensitivity,
                      estimated_value)
        stats_parameters[stat] = statistics

    if len(zero_sigmas) > 0:
        # If you want a counter with no noise, try using 1e-6 instead
        logging.error("sensitivity for {} is zero, calculated sigmas will be zero for all statistics"
                        .format(summarise_list(zero_sigmas)))

    # calculate the noise allocations
    # and update the structure with defaults, if not already present
    epsilons, sigmas, sigma_ratio = \
        get_opt_privacy_allocation(noise['epsilon'],
                                   noise['delta'],
                                   stats_parameters,
                                   excess_noise_ratio,
                                   sigma_tol=noise.setdefault(
                                                 'sigma_tolerance',
                                                 DEFAULT_SIGMA_TOLERANCE),
                                   epsilon_tol=noise.setdefault(
                                                 'epsilon_tolerance',
                                                 DEFAULT_EPSILON_TOLERANCE),
                                   sigma_ratio_tol=noise.setdefault(
                                                 'sigma_ratio_tolerance',
                                                 DEFAULT_SIGMA_RATIO_TOLERANCE)
                               )
    # update the structure with the results
    noise['sigma_ratio'] = sigma_ratio
    for stat in counters:
        counters[stat]['epsilon'] = epsilons[stat]
        counters[stat]['sigma'] = sigmas[stat]
        noise_ratio = get_expected_noise_ratio(
            excess_noise_ratio,
            counters[stat]['sigma'],
            counters[stat]['estimated_value'])
        counters[stat]['expected_noise_ratio'] = noise_ratio
    if sanity_check is not None:
        counters[sanity_check] = get_sanity_check_counter()
    return noise_parameters

def get_noise_allocation_stats(epsilon, delta, stats_parameters,
                               excess_noise_ratio,
                               sigma_tol=None,
                               epsilon_tol=None,
                               sigma_ratio_tol=None,
                               sanity_check=DEFAULT_DUMMY_COUNTER_NAME,
                               circuit_sample_rate=1.0):
    '''
    Like get_noise_allocation, but uses the structure stats_parameters:
    - 'CounterName': multiple
        statistics: (sensitivity float in, estimated_value float in)
    And the variables:
        epsilon: float in
        delta: float in
        excess_noise_ratio: float in
        sigma_tolerance: float in optional default None (1e-6)
        epsilon_tolerance: float in optional default None (1e-15)
        sigma_ratio_tolerance: float in optional default None (1e-6)
    And calls get_noise_allocation, to return the result:
    privacy:
        (as in get_noise_allocation)
    counters:
        'CounterName': multiple
            sensitivity: float out
            estimated_value: float out
            (remainder as in get_noise_allocation)
    A sanity check counter is added by default, see get_noise_allocation() for
    details.
    This adapter is used as part of the unit tests. It should produce the same
    results as calling get_opt_privacy_allocation() directly (but structured in
    the format used by get_noise_allocation()).
    '''
    noise_parameters = {}
    # construct the noise part
    noise_parameters['privacy'] = {}
    noise_parameters['privacy']['epsilon'] = epsilon
    noise_parameters['privacy']['delta'] = delta
    noise_parameters['privacy']['excess_noise_ratio'] = excess_noise_ratio
    if sigma_tol is not None:
        noise_parameters['privacy']['sigma_tolerance'] = sigma_tol
    if epsilon_tol is not None:
        noise_parameters['privacy']['epsilon_tolerance'] = epsilon_tol
    if sigma_ratio_tol is not None:
        noise_parameters['privacy']['sigma_ratio_tolerance'] = sigma_ratio_tol
    # construct the counter part
    noise_parameters['counters'] = {}
    for stat in stats_parameters:
        counter = {}
        (sensitivity, estimated_value) = stats_parameters[stat]
        counter['sensitivity'] = sensitivity
        counter['estimated_value'] = estimated_value
        noise_parameters['counters'][stat] = counter
    return get_noise_allocation(noise_parameters, sanity_check=sanity_check,
                                circuit_sample_rate=circuit_sample_rate)

def compare_noise_allocation(epsilon, delta, stats_parameters,
                             excess_noise_ratio,
                             sigma_tol=DEFAULT_SIGMA_TOLERANCE,
                             epsilon_tol=DEFAULT_EPSILON_TOLERANCE,
                             sigma_ratio_tol=DEFAULT_SIGMA_RATIO_TOLERANCE,
                             sanity_check=DEFAULT_DUMMY_COUNTER_NAME):
    '''
    Call get_opt_privacy_allocation() and get_noise_allocation_stats(),
    and assert that the results are equivalent.
    '''
    # Call the base function
    epsilons, sigmas, sigma_ratio =\
        get_opt_privacy_allocation(epsilon, delta, stats_parameters,
                                   excess_noise_ratio,
                                   sigma_tol=sigma_tol,
                                   epsilon_tol=epsilon_tol,
                                   sigma_ratio_tol=sigma_ratio_tol)
    # Add the sanity check counter
    if sanity_check is not None:
        sigmas[sanity_check] = get_sanity_check_counter()['sigma']
        epsilons[sanity_check] = get_sanity_check_counter()['epsilon']
    # Call the high-level function
    noise_parameters =\
        get_noise_allocation_stats(epsilon, delta, stats_parameters,
                                   excess_noise_ratio,
                                   sigma_tol=sigma_tol,
                                   epsilon_tol=epsilon_tol,
                                   sigma_ratio_tol=sigma_ratio_tol,
                                   sanity_check=sanity_check)
    # assert that the results and calculated values are equivalent
    # base results
    # double equality fails here, so we compare string representations
    # instead: see #260 and #69
    # and yes, this is a terrible hack
    assert str(noise_parameters['privacy']['sigma_ratio']) == str(sigma_ratio)
    np_excess_noise_ratio = noise_parameters['privacy']['excess_noise_ratio']
    for stat in noise_parameters['counters']:
        counter = noise_parameters['counters'][stat]
        assert str(epsilons[stat]) == str(counter['epsilon'])
        assert str(sigmas[stat]) == str(counter['sigma'])
        # rearranged values
        if stat == sanity_check:
            sc = get_sanity_check_counter()
            base_sensitivity = sc['sensitivity']
            base_estimated_value = sc['estimated_value']
        else:
            (base_sensitivity, base_estimated_value) = stats_parameters[stat]
        np_sensitivity = counter['sensitivity']
        np_estimated_value = counter['estimated_value']
        assert str(base_sensitivity) == str(np_sensitivity)
        assert str(base_estimated_value) == str(np_estimated_value)
        # calculated values
        base_noise_ratio = get_expected_noise_ratio(excess_noise_ratio,
                                                    sigmas[stat],
                                                    base_estimated_value)
        np_noise_ratio = get_expected_noise_ratio(np_excess_noise_ratio,
                                                  counter['sigma'],
                                                  counter['estimated_value'])
        calc_noise_ratio = counter['expected_noise_ratio']
        assert str(base_noise_ratio) == str(np_noise_ratio)
        assert str(base_noise_ratio) == str(calc_noise_ratio)

def print_privacy_allocation(stats_parameters, sigmas, epsilons, excess_noise_ratio):
    '''
    Print information about sigmas and noise ratios for each statistic.
    '''
    # print epsilons sorted by epsilon allocation
    epsilons_sorted = epsilons.keys()
    epsilons_sorted.sort(key = lambda x: epsilons[x], reverse = True)
    print('Epsilon values\n-----')
    for param in epsilons_sorted:
        print('{}: {}'.format(param, epsilons[param]))
    # print equalized ratio of sigma to expected value
    for param, stats in stats_parameters.iteritems():
        ratio = get_expected_noise_ratio(excess_noise_ratio,
                                         sigmas[param],
                                         stats[1])
        print('Ratio of sigma value to expected value: {}'.format(ratio))
        break

    print ""

    # print allocation of privacy budget
    sigma_params_sorted = sigmas.keys()
    # add dummy counter for full counters.noise.yaml
    dummy_counter = DEFAULT_DUMMY_COUNTER_NAME
    sigma_params_sorted.append(dummy_counter)
    sigmas[dummy_counter] = get_sanity_check_counter()['sigma']
    epsilons[dummy_counter] = get_sanity_check_counter()['epsilon']
    sigma_params_sorted.sort()
    print('Sigma values\n-----')
    print('counters:')
    for param in sigma_params_sorted:
        sigma = sigmas[param]
        print('    {}:'.format(param))
        print('        sigma: {:4f}'.format(sigma))

    print ""
