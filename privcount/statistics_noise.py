import logging
import math
import scipy.stats
import yaml

DEFAULT_SIGMA_TOLERANCE = 1e-6
DEFAULT_EPSILON_TOLERANCE = 1e-15
DEFAULT_SIGMA_RATIO_TOLERANCE = 1e-6

DEFAULT_DUMMY_COUNTER_NAME = 'SanityCheck'

def satisfies_dp(sensitivity, epsilon, delta, std):
    '''
    Return True if (epsilon, delta)-differential privacy is satisfied.
    '''
    # find lowest value at which epsilon differential-privacy is satisfied
    lower_x = -(float(epsilon) * (std**2.0) / sensitivity) + sensitivity/2.0
    # determine lower tail probability of normal distribution w/ mean of zero
    lower_tail_prob = scipy.stats.norm.cdf(lower_x, 0, std)
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
#        print('Interval search lower: {}'.format(lower_bound))
#        print('Interval search upper: {}'.format(upper_bound))
        if ((upper_bound - lower_bound) < tol):
            if return_true is True:
                return upper_bound
            else:
                return lower_bound
        midpoint = float(upper_bound + lower_bound) / 2
        midpoint_val = fn(midpoint)
#        print('Midpoint: {}'.format(midpoint))
#        print('Midpoint val: {}'.format(midpoint_val))
        if (midpoint_val is False):
            lower_bound = midpoint
        else:
            upper_bound = midpoint

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
    for param, (sensitivity, val) in stats_parameters.iteritems():
        opt_sigma = get_sigma(excess_noise_ratio, opt_sigma_ratio, val)
        # Check if the sigma is too small
        if opt_sigma == 0.0:
            pass
        elif opt_sigma < DEFAULT_SIGMA_TOLERANCE:
            logging.warning("sigma {} for {} is less than the sigma tolerance {}, the calculated value may be inaccurate and may vary each time it is calculated"
                            .format(opt_sigma, param, sigma_tol))

        opt_sigmas[param] = opt_sigma

    return (opt_epsilons, opt_sigmas, opt_sigma_ratio)

def get_sanity_check_counter():
    '''
    Provide a dictionary with the standard sanity check counter values.
    It is typically used like:
        counters['SanityCheck'] = get_sanity_check_counter()
    All of these values are unused, except for:
        bins:
            - [0.0, inf] (a long counter is appended by SecureCounters,
                           it should only ever have blinding values added)
        estimated_value: 0.0 (TODO: used for checking if stats have changed)
        sigma: 0.0 (used for adding noise, 0.0 means no noise is added)
    '''
    sanity_check = {}
    sanity_check['bins'] = [0.0, float('inf')]
    sanity_check['sensitivity'] = 0.0
    sanity_check['estimated_value'] = 0.0
    sanity_check['sigma'] = 0.0
    sanity_check['epsilon'] = 0.0
    sanity_check['expected_noise_ratio'] = 0.0
    return sanity_check

def get_noise_allocation(noise_parameters,
                         sanity_check=DEFAULT_DUMMY_COUNTER_NAME):
    '''
    An adapter which wraps get_opt_privacy_allocation, extracting the
    parameters from the noise_parameters data structure, and updating
    noise_parameters with the calculated values.
    If sanity_check is not None, adds a sanity check counter to the result,
    with the counter name supplied in sanity_check, and values created using
    get_sanity_check_counter().
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
    # extract the top-level structures
    noise = noise_parameters['privacy']
    counters = noise_parameters['counters']
    excess_noise_ratio = noise['excess_noise_ratio']
    # rearrange the counter values, and produce the parameter-only structure
    stats_parameters = {}
    for stat in counters:
        statistics = (counters[stat]['sensitivity'],
                      counters[stat]['estimated_value'])
        stats_parameters[stat] = statistics
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
                               sanity_check=DEFAULT_DUMMY_COUNTER_NAME):
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
    return get_noise_allocation(noise_parameters, sanity_check=sanity_check)

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
    assert noise_parameters['privacy']['sigma_ratio'] == sigma_ratio
    np_excess_noise_ratio = noise_parameters['privacy']['excess_noise_ratio']
    for stat in noise_parameters['counters']:
        counter = noise_parameters['counters'][stat]
        assert epsilons[stat] == counter['epsilon']
        assert sigmas[stat] == counter['sigma']
        # rearranged values
        if stat == sanity_check:
            sc = get_sanity_check_counter()
            base_sensitivity = sc['sensitivity']
            base_estimated_value = sc['estimated_value']
        else:
            (base_sensitivity, base_estimated_value) = stats_parameters[stat]
        np_sensitivity = counter['sensitivity']
        np_estimated_value = counter['estimated_value']
        assert base_sensitivity == np_sensitivity
        assert base_estimated_value == np_estimated_value
        # calculated values
        base_noise_ratio = get_expected_noise_ratio(excess_noise_ratio,
                                                    sigmas[stat],
                                                    base_estimated_value)
        np_noise_ratio = get_expected_noise_ratio(np_excess_noise_ratio,
                                                  counter['sigma'],
                                                  counter['estimated_value'])
        calc_noise_ratio = counter['expected_noise_ratio']
        assert base_noise_ratio == np_noise_ratio
        assert base_noise_ratio == calc_noise_ratio

# privacy sensitivity
sensitivity_client_ips_per_slice = 1
sensitivity_client_ips_duration = 60*60*24 # duration to cover IP for
sensitivity_connections = 12 # one connection per hour for 12 hours
sensitivity_circuits = 6*24+2 # constant use for 24 hours: two pre-emptive circuits plus six circuits per hour (due to a circuit lifetime of 10 minutes)
sensitivity_web_circuits = 6*24+2 # constant use for 24 hours
sensitivity_interactive_circuits = 20 # 2 per hour for 10 hours
sensitivity_p2p_circuits = 6*24+2 # constant use for 24 hours
sensitivity_other_circuits = 6*24+2 # constant use for 24 hours
sensitivity_streams = 150 * 200 # max of likely applications (viz. Web)
sensitivity_web_streams = 150 * 200 # number of Web pages (200) * number of objects per page (150) ?
sensitivity_interactive_streams = 20 # 2 per hour for 10 hours
sensitivity_p2p_streams = 2*40 # "Analyzing and Improving a BitTorrent Network's Performance Mechanisms" (INFOCOM 2006) states (Sec. 2) that a "new node attempts to establish connections to about 40 existing nodes, which then become its neighbors". We double this number to allow for node churn.
sensitivity_other_streams = sensitivity_circuits-2 # not sure what other circuits might be used for - allow a user to have one per circuit privately
sensitivity_web_kibytes = 10*1024 # 10 MiB, should cover nearly all web pages (see total transfer size per page distribution at <http://httparchive.org/interesting.php>)
sensitivity_interactive_kibytes = 10*1024 # use at least the Web amount
sensitivity_p2p_kibytes = 10*1024 # use at least the Web amount
sensitivity_other_kibytes = 10*1024 # use at least the Web amount
sensitivity_kibytes = max(sensitivity_web_kibytes,
    sensitivity_interactive_kibytes, sensitivity_p2p_kibytes,
    sensitivity_other_kibytes)

# measurement parameters
initial_epoch_length = 1*24*60*60 # length of initial measurement epoch in seconds
initial_epoch_days = float(initial_epoch_length) / (60*60*24)
p2p_initial_epoch_length = 1*24*60*60
p2p_initial_epoch_days = float(p2p_initial_epoch_length) / (60*60*24)
epoch_length = 4*24*60*60 # epoch length in seconds
epoch_days = float(epoch_length) / (60*60*24)
slice_length = 10*60 # time slice for sensitive data in seconds
num_relay_machines = 3 # number of relays collecting statistics

# expected statistics
## estimates from extrainfo descriptors from 7 relays on April 21st ##
# taken from tortraffic.git/statistics/extrainfo_estimates/20160421/output2_with_p2p.txt
extrainfo_num_streams_per_day = 54400816.06
extrainfo_num_web_streams_per_day = 51895129.8694
extrainfo_num_interactive_streams_per_day = 48010.8145628
extrainfo_num_p2p_streams_per_day = 43961.9269942
extrainfo_num_other_streams_per_day = 510781.687051 + 1946893.68892 # all streams with an "other" port + all streams without a port indicated in the stats (aka what estimate_extrainfo_stats.py calls "unclassified" streams aka what the extrainfo docs call "other")
extrainfo_num_circuits_per_day = 12154275.4265
extrainfo_num_ips_per_day = 306137.102559
extrainfo_num_kibytes_per_day = 3794763361.66 + 192927246 # take sum of read and written KiB, which for exits are counted only on the exit-dest cxn
extrainfo_num_interactive_kibytes_per_day = 12750231.9211 + 867648.146583 # take sum of read and written KiB, which for exits are counted only on the exit-dest cxn
extrainfo_num_other_kibytes_per_day = 106438782.369+289369179.715 + 11159489.3116+76396227.9758 # take sum of read and written KiB for what estimate_extrainfo_stats.py calls "other" and "unclassified", which for exits are counted only on the exit-dest cxn
extrainfo_num_web_kibytes_per_day = 3380782907.61 + 102015820.891 # take sum of read and written KiB, which for exits are counted only on the exit-dest cxn
extrainfo_num_p2p_kibytes_per_day = 5422260.04588 + 2488060.04996

# taken from p2p initial data collection 6/16/16-6/17/16
num_p2p_streams_per_day = 966004.0/2.0
num_streams_per_day = 58157663.0/2
num_p2p_circuits_per_day = 571087.0/2
num_circuits_per_day = 8131540.0/2
num_p2p_kibytes_per_day = 93522051.0/2

# taken from initial data collection on 4/29/16 and
num_web_streams_per_day = 31335162.0
num_interactive_streams_per_day = 8905.0
num_other_streams_per_day = 1863013.0
num_active_circuits_per_day = 1699756.0
num_inactive_circuits_per_day = 1241706.0
num_web_circuits_per_day = 1415683.0
num_interactive_circuits_per_day = 6645.0
num_other_circuits_per_day = 490744.0
#num_ips_slices_per_day = 79505.0 # num ip-slices, i.e., sum over time slices of unique ips per slice # removed in favor of more accurate estimate from later entry collection
num_kibytes_per_day = 2207373238.0
num_interactive_kibytes_per_day = 6300144.0
num_other_kibytes_per_day = 375853584.0
num_web_kibytes_per_day = 2394910504.0

# taken from two-day collection of guard-only statistics starting 8/8/16
num_ips_slices_per_day = 297021.0/2
num_active_ips_slices_per_day = 201066.0/2
num_inactive_ips_slices_per_day = 107108.0/2
num_connections_per_day = 147162.0/2

# p2p initial statistics (needed to get estimates for other initial rounds)
p2p_initial_stats_parameters = {\
    'CircuitsAll' : (sensitivity_circuits,
        extrainfo_num_circuits_per_day * p2p_initial_epoch_days),
    'CircuitsP2P' : (sensitivity_p2p_circuits,
         (float(extrainfo_num_p2p_streams_per_day)/2.0)*p2p_initial_epoch_days), # est 2 streams/circuit under the logic that most BitTorrent peers will have at most that many simultaneous downloads from different peers in a given circuit lifetime (i.e. 10 min) or will complete at most that many successive piece/sub-piece downloads in the circuit lifetime
    'StreamsAll' : (sensitivity_streams,
        extrainfo_num_streams_per_day * p2p_initial_epoch_days),
    'StreamsP2P' : (sensitivity_p2p_streams,
        extrainfo_num_p2p_streams_per_day * p2p_initial_epoch_days),
    'StreamBytesP2P' : (sensitivity_p2p_kibytes,
        extrainfo_num_p2p_kibytes_per_day * p2p_initial_epoch_days)
}

# initial statistics for basic data exploration (uses mostly extrainfo estimates but also some hard-to-estimate values from the P2P initial collection
initial_stats_parameters = {\
    'CircuitsAll' : (sensitivity_circuits,
        num_circuits_per_day * initial_epoch_days),
    'CircuitsActive' : (sensitivity_circuits,
        0.5*num_circuits_per_day * initial_epoch_days),
    'CircuitsInactive' : (sensitivity_circuits,
        0.5*num_circuits_per_day * initial_epoch_days),
    'CircuitsInteractive' : (sensitivity_interactive_circuits,
        (float(extrainfo_num_interactive_streams_per_day) / 2) * initial_epoch_days), # est 2 streams / interactive circuit
    'CircuitsOther' : (sensitivity_other_circuits,
        extrainfo_num_circuits_per_day * (float(extrainfo_num_other_streams_per_day)/extrainfo_num_streams_per_day) * initial_epoch_days), # est same fraction of other circuits as other streams
    'CircuitsP2P' : (sensitivity_p2p_circuits,
         (float(num_p2p_circuits_per_day))*initial_epoch_days),
    'CircuitsWeb' : (sensitivity_web_circuits, # est 50 streams / web circ
        (float(extrainfo_num_web_streams_per_day) / 50) * initial_epoch_days),
    'ClientIPsUnique' : (\
        sensitivity_client_ips_per_slice * float(sensitivity_client_ips_duration)/slice_length,
        extrainfo_num_ips_per_day * initial_epoch_days),
    'ClientIPsActive' : (sensitivity_client_ips_per_slice *\
        float(sensitivity_client_ips_duration)/slice_length,
        0.01 * extrainfo_num_ips_per_day * initial_epoch_days), # est 1/100 of clients with circuits in a time slice ever use those circuits
    'ClientIPsInactive' : (sensitivity_client_ips_per_slice *\
        float(sensitivity_client_ips_duration)/slice_length,
        0.99 * extrainfo_num_ips_per_day * initial_epoch_days), # est 99/100 of clients with circuits in a time slice don't use those circuits,),
    'StreamBytesAll' : (sensitivity_kibytes, extrainfo_num_kibytes_per_day * initial_epoch_days),
    'StreamBytesInteractive' : (sensitivity_interactive_kibytes,
        extrainfo_num_interactive_kibytes_per_day * initial_epoch_days),
    'StreamBytesOther' : (sensitivity_other_kibytes,
        extrainfo_num_other_kibytes_per_day * initial_epoch_days),
    'StreamBytesP2P' : (sensitivity_p2p_kibytes,
        num_p2p_kibytes_per_day * initial_epoch_days),
    'StreamBytesWeb' : (sensitivity_web_kibytes,
        extrainfo_num_web_kibytes_per_day * initial_epoch_days),
    'StreamsAll' : (sensitivity_streams,
        num_streams_per_day * p2p_initial_epoch_days),
    'StreamsInteractive' : (sensitivity_interactive_streams,
        extrainfo_num_interactive_streams_per_day * initial_epoch_days),
    'StreamsOther' : (sensitivity_other_streams,
        extrainfo_num_other_streams_per_day * initial_epoch_days),
    'StreamsP2P' : (sensitivity_p2p_streams,
        num_p2p_streams_per_day * initial_epoch_days),
    'StreamsWeb' : (sensitivity_streams,
        extrainfo_num_web_streams_per_day * initial_epoch_days)
}

# name some histogram parameters that will be reused
circuit_histogram_parameters = (2*sensitivity_circuits,
    num_circuits_per_day * epoch_days)
stream_histogram_parameters = (2 * sensitivity_streams, num_streams_per_day *\
    epoch_days)
web_stream_histogram_parameters = (2 * sensitivity_web_streams, num_web_streams_per_day *\
    epoch_days)
interactive_stream_histogram_parameters = (2 * sensitivity_interactive_streams,
    num_interactive_streams_per_day * epoch_days)
# removing P2P class
#p2p_stream_histogram_parameters = (2 * sensitivity_p2p_streams,
#    num_p2p_streams_per_day * epoch_days)
other_stream_histogram_parameters = (2 * sensitivity_other_streams,
    num_other_streams_per_day * epoch_days)

# map statistics name to tuple of (maximum distance, expected value)
# note histograms contain two factor because a changed entry reduces one bucket and increases another
stats_parameters = {\
    ### entry statistics ###
    ## counts ##
    'ClientIPsUnique' : (\
        sensitivity_client_ips_per_slice * float(sensitivity_client_ips_duration)/slice_length,
        num_ips_slices_per_day * epoch_days),
    'ClientIPsActive' : (sensitivity_client_ips_per_slice *\
        float(sensitivity_client_ips_duration)/slice_length,
        num_active_ips_slices_per_day * epoch_days), # used to estimate with 0.1 * num_ips_slices_per_day instead of num_active_ips_slices_per_day
    'ClientIPsInactive' :(sensitivity_client_ips_per_slice *\
        float(sensitivity_client_ips_duration)/slice_length,
        num_inactive_ips_slices_per_day * epoch_days),
    'ConnectionsAll' : (sensitivity_connections, num_connections_per_day * epoch_days), # used to use num_ips_slices_per_day instead of num_connections_per_day w/ an est. of 1 cxn per IP per day
    ####
    ## histograms ##
# removed due to low utility and complication of counting circuits at both guards and exits
#    'CircuitCellsIn' : circuit_histogram_parameters,
#    'CircuitCellsOut' : circuit_histogram_parameters,
#    'CircuitCellsRatio' : circuit_histogram_parameters,
    ####
    ######

    ### exit statistics ###
    ## counts ##
    'CircuitsActive' : (sensitivity_circuits, num_active_circuits_per_day * epoch_days),
    'CircuitsInactive' : (sensitivity_circuits, num_inactive_circuits_per_day * epoch_days),
# removing interactive stats due to low volume
#    'CircuitsInteractive' : (sensitivity_interactive_circuits,
#        num_interactive_circuits_per_day * epoch_days),
    'CircuitsOther' : (sensitivity_other_circuits, num_other_circuits_per_day * epoch_days),
# removing P2P class
#    'CircuitsP2P' : (sensitivity_p2p_circuits, num_p2p_circuits_per_day * epoch_days),
    'CircuitsWeb' : (sensitivity_web_circuits, num_web_circuits_per_day * epoch_days),
    'StreamBytesAll' : (sensitivity_kibytes, num_kibytes_per_day * epoch_days),
# removing interactive stats due to low volume
#    'StreamBytesInteractive' : (sensitivity_interactive_kibytes,
#        num_interactive_kibytes_per_day * epoch_days),
    'StreamBytesOther' : (sensitivity_other_kibytes, num_other_kibytes_per_day * epoch_days),
    'StreamBytesWeb' : (sensitivity_web_kibytes, num_web_kibytes_per_day * epoch_days),
    'StreamsAll' : (sensitivity_streams, num_streams_per_day * epoch_days),
# removing interactive stats due to low volume
#    'StreamsInteractive' : (sensitivity_interactive_streams,
#        num_interactive_streams_per_day * epoch_days),
    'StreamsOther' : (sensitivity_other_streams, num_other_streams_per_day * epoch_days),
# removing P2P class
#    'StreamsP2P' : (sensitivity_p2p_streams, num_p2p_streams_per_day * epoch_days),
    'StreamsWeb' : (sensitivity_streams, num_web_streams_per_day * epoch_days),
    ####

    ## histograms ##
    'CircuitInterStreamCreationTime' : stream_histogram_parameters,
    'CircuitInterStreamCreationTimeOther' : other_stream_histogram_parameters,
    'CircuitInterStreamCreationTimeWeb' : web_stream_histogram_parameters,
    'CircuitLifeTimeAll' : circuit_histogram_parameters,
    'CircuitLifeTimeActive' : (2*sensitivity_circuits, num_active_circuits_per_day * epoch_days),
    'CircuitLifeTimeInactive' : (2*sensitivity_circuits, num_inactive_circuits_per_day * epoch_days),
    'CircuitStreamsAll' : circuit_histogram_parameters,
    'CircuitStreamsOther' : (2 * sensitivity_other_circuits,
        num_other_circuits_per_day * epoch_days),
    'CircuitStreamsWeb' : (2 * sensitivity_web_circuits, num_web_circuits_per_day * epoch_days),
# removing interactive stats due to low volume
#    'CircuitStreamsInteractive' : (2 * sensitivity_interactive_circuits,
#        num_interactive_circuits_per_day * epoch_days),
# removing P2P class
#    'CircuitStreamsP2P' : (2 * sensitivity_p2p_circuits, num_p2p_circuits_per_day * epoch_days),
    'StreamBytesInAll' : stream_histogram_parameters,
    'StreamBytesInOther' : other_stream_histogram_parameters,
    'StreamBytesInWeb' : web_stream_histogram_parameters,
# removing interactive stats due to low volume
#    'StreamBytesInInteractive' : interactive_stream_histogram_parameters,
# removing P2P class
#    'StreamBytesInP2P' : p2p_stream_histogram_parameters,
    'StreamBytesOutAll' : stream_histogram_parameters,
    'StreamBytesOutOther' : other_stream_histogram_parameters,
    'StreamBytesOutWeb' : web_stream_histogram_parameters,
# removing interactive stats due to low volume
#    'StreamBytesOutInteractive' : interactive_stream_histogram_parameters,
# removing P2P class
#    'StreamBytesOutP2P' : p2p_stream_histogram_parameters,
    'StreamBytesRatioAll' : stream_histogram_parameters,
    'StreamBytesRatioOther' : other_stream_histogram_parameters,
    'StreamBytesRatioWeb' : web_stream_histogram_parameters
# removing interactive stats due to low volume
#    'StreamBytesRatioInteractive':interactive_stream_histogram_parameters,
# removing P2P class
#    'StreamBytesRatioP2P' : p2p_stream_histogram_parameters,
    ####
    ######
}

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

if __name__ == '__main__':
    epsilon = 0.3
    delta = 1e-3
    excess_noise_ratio = num_relay_machines # factor by which noise is expanded to allow for malicious relays
    sigma_tol=DEFAULT_SIGMA_TOLERANCE
    epsilon_tol=DEFAULT_EPSILON_TOLERANCE
    sigma_ratio_tol=DEFAULT_SIGMA_RATIO_TOLERANCE

    ## P2P (and other added) initial statistics ##
    p2p_initial_epsilons, p2p_initial_sigmas, p2p_initial_sigma_ratio =\
        get_opt_privacy_allocation(epsilon, delta, p2p_initial_stats_parameters,
            excess_noise_ratio, sigma_tol=sigma_tol, epsilon_tol=epsilon_tol,
            sigma_ratio_tol=sigma_ratio_tol)
    # print information about initial statistics noise
    print('* P2P initial statistics *\n')
    print_privacy_allocation(p2p_initial_stats_parameters, p2p_initial_sigmas,
        p2p_initial_epsilons, excess_noise_ratio)
    compare_noise_allocation(epsilon, delta, p2p_initial_stats_parameters,
                             excess_noise_ratio,
                             sigma_tol=sigma_tol,
                             epsilon_tol=epsilon_tol,
                             sigma_ratio_tol=sigma_ratio_tol,
                             sanity_check=DEFAULT_DUMMY_COUNTER_NAME)
    p2p_initial_noise_parameters =\
        get_noise_allocation_stats(epsilon, delta,
                                   p2p_initial_stats_parameters,
                                   excess_noise_ratio,
                                   sigma_tol=sigma_tol,
                                   epsilon_tol=epsilon_tol,
                                   sigma_ratio_tol=sigma_ratio_tol,
                                   sanity_check=DEFAULT_DUMMY_COUNTER_NAME)
    print('\nnoise config\n')
    print yaml.dump(p2p_initial_noise_parameters, default_flow_style=False)
    ####

    ## initial statistics ##
    # get optimal noise allocation for initial statistics
    (initial_epsilons, initial_sigmas, initial_sigma_ratio) =  get_opt_privacy_allocation(epsilon,
        delta, initial_stats_parameters, excess_noise_ratio, sigma_tol=sigma_tol,
        epsilon_tol=epsilon_tol, sigma_ratio_tol=sigma_ratio_tol)
    # print information about initial statistics noise
    print('\n* Initial statistics *\n')
    print_privacy_allocation(initial_stats_parameters, initial_sigmas,
        initial_epsilons, excess_noise_ratio)
    compare_noise_allocation(epsilon, delta, initial_stats_parameters,
                             excess_noise_ratio,
                             sigma_tol=sigma_tol,
                             epsilon_tol=epsilon_tol,
                             sigma_ratio_tol=sigma_ratio_tol,
                             sanity_check=DEFAULT_DUMMY_COUNTER_NAME)
    initial_noise_parameters =\
        get_noise_allocation_stats(epsilon, delta, initial_stats_parameters,
                                   excess_noise_ratio,
                                   sigma_tol=sigma_tol,
                                   epsilon_tol=epsilon_tol,
                                   sigma_ratio_tol=sigma_ratio_tol,
                                   sanity_check=DEFAULT_DUMMY_COUNTER_NAME)
    print('\nnoise config\n')
    print yaml.dump(initial_noise_parameters, default_flow_style=False)
    ####

    ## full statistics ##
    # get optimal noise allocation for full statistics
    full_epsilons, full_sigmas, full_sigma_ratio = get_opt_privacy_allocation(epsilon, delta,
        stats_parameters, excess_noise_ratio, sigma_tol=sigma_tol, epsilon_tol=epsilon_tol,
        sigma_ratio_tol=sigma_ratio_tol)
    # print information about full statistics noise
    print('\n* Full statistics *\n')
    print_privacy_allocation(stats_parameters, full_sigmas, full_epsilons, excess_noise_ratio)
    compare_noise_allocation(epsilon, delta, stats_parameters,
                             excess_noise_ratio,
                             sigma_tol=sigma_tol,
                             epsilon_tol=epsilon_tol,
                             sigma_ratio_tol=sigma_ratio_tol,
                             sanity_check=DEFAULT_DUMMY_COUNTER_NAME)
    noise_parameters =\
        get_noise_allocation_stats(epsilon, delta, stats_parameters,
                                   excess_noise_ratio,
                                   sigma_tol=sigma_tol,
                                   epsilon_tol=epsilon_tol,
                                   sigma_ratio_tol=sigma_ratio_tol,
                                   sanity_check=DEFAULT_DUMMY_COUNTER_NAME)
    print('\nnoise config\n')
    print yaml.dump(noise_parameters, default_flow_style=False)
    # Debug output
    if False:
        print "Inputs:"
        print "epsilon"
        print epsilon
        print "delta"
        print delta
        print "stats_parameters"
        print stats_parameters
        print "excess_noise_ratio"
        print excess_noise_ratio
        print "sigma_tol"
        print sigma_tol
        print "epsilon_tol"
        print epsilon_tol
        print "sigma_ratio_tol"
        print sigma_ratio_tol
        print "Outputs:"
        print "full_epsilons"
        print full_epsilons
        print "full_sigmas"
        print full_sigmas
        print "full_sigma_ratio"
        print full_sigma_ratio
        print "noise_parameters"
        print noise_parameters
    ####
