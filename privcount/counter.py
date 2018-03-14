'''
Created on Dec 6, 2016

@author: teor

See LICENSE for licensing information
'''

import logging
import sys

from random import SystemRandom
from copy import deepcopy
from math import sqrt, isnan

from privcount.config import _extra_keys, _common_keys
from privcount.log import format_period, format_elapsed_time_since, format_delay_time_until, summarise_list

DEFAULT_SIGMA_TOLERANCE = 1e-6
DEFAULT_EPSILON_TOLERANCE = 1e-15
DEFAULT_SIGMA_RATIO_TOLERANCE = 1e-6

DEFAULT_DUMMY_COUNTER_NAME = 'ZeroCount'
# The label used for the default noise weight for testing
# Labels are typically data collector relay fingerprints
DEFAULT_NOISE_WEIGHT_NAME = '*'

def counter_modulus():
    '''
    The hard-coded modulus value for a blinded counter
    Blinded counters are unsigned
    In PrivCount, this does not have to be prime, and there is no need for it
    to be configurable
    All PrivCount counters should use unlimited-length Python longs, so that
    counter_modulus can exceed 64 bits, the size of a native C long
    '''
    # PrivCount counters are limited by the modulus, so it needs to be large
    # Here's an over-estimate of PrivCount's capacity:
    # In 2016, Tor traffic was 75 Gbits, or ~2**34 bytes per second
    # (In 2015, Internet traffic was 230 Tbits, or ~2**43 bytes per second)
    # Tor traffic might grow by 2**10 while PrivCount is in use
    # A year has ~2**25 seconds
    # PrivCount counters overflow at modulus/2
    # 2**34 * 2**10 * 2**25 * 2 = 2**70
    # Using modulus > 2**64 also ensures PrivCount is unlimited-integer clean
    # and that it can handle longs that just happen to be integers
    # (1 in 2**6 blinding factors are less than 2**64)
    return 2L**70L
    # historical q values
    #return 2147483647L
    #return 999999999959L
    # modulus was limited to 2**64 when sample() only unpacked 8 bytes
    #return 2L**64L

def min_blinded_counter_value():
    '''
    The hard-coded minimum value for a blinded counter
    Blinded counters are unsigned
    Always zero
    '''
    return 0L

def max_blinded_counter_value():
    '''
    The hard-coded maximum value for a blinded counter
    Blinded counters are unsigned
    '''
    return counter_modulus() - 1L

def min_tally_counter_value():
    '''
    The hard-coded minimum value for a tallied counter
    Tallied counters are signed, to allow for negative noise
    '''
    return adjust_count_signed((counter_modulus() + 1L)//2L,
                               counter_modulus())

def max_tally_counter_value():
    '''
    The hard-coded maximum value for a tallied counter
    Tallied counters are signed, to allow for negative noise
    '''
    return adjust_count_signed((counter_modulus() + 1L)//2L - 1L,
                               counter_modulus())

def add_counter_limits_to_config(config):
    '''
    Add the hard-coded counter limits to a deep copy of the config dictionary
    Returns the modified deep copy of the config dictionary
    '''
    assert config is not None
    config = deepcopy(config)
    # call this modulus so it sorts near the other values
    config['modulus'] = counter_modulus()
    config['min_blinded_counter_value'] = min_blinded_counter_value()
    config['max_blinded_counter_value'] = max_blinded_counter_value()
    config['min_tally_counter_value'] = min_tally_counter_value()
    config['max_tally_counter_value'] = max_tally_counter_value()
    return config

MAX_DC_COUNT = 10**6

def check_dc_threshold(dc_threshold, description="threshold"):
    '''
    Check that dc_threshold is a valid dc threshold.
    DC thresholds must be positive non-zero, and less than or equal to
    MAX_DC_COUNT.
    Returns True if the dc threshold is valid.
    Logs a specific warning using description and returns False if it is not.
    '''
    if dc_threshold <= 0:
        logging.warning("Data collector {} must be at least 1, was {}"
                        .format(description, dc_threshold))
        return False
    if dc_threshold > MAX_DC_COUNT:
        logging.warning("Data collector {} can be at most {}, was {}"
                        .format(description, MAX_DC_COUNT, dc_threshold))
        return False
    return True

def check_noise_weight_value(noise_weight_value, description="value"):
    '''
    Check that noise_weight_value is a valid noise weight.
    Noise weights must be positive and less than or equal to the maximum
    tallied counter value.
    Returns True if the noise weight value is valid.
    Logs a specific warning using description, and returns False if it is not.
    '''
    if noise_weight_value < 0.0:
        logging.warning("Noise weight {} must be positive, was {}".format(
                description, noise_weight_value))
        return False
    if noise_weight_value > max_tally_counter_value():
        logging.warning("Noise weight {} can be at most {}, was {}".format(
                description, max_tally_counter_value(), noise_weight_value))
        return False
    return True

def check_noise_weight_sum(noise_weight_sum, description="sum"):
    '''
    Check that noise_weight_sum is a valid summed noise weight.
    Noise weight sums must pass check_noise_weight_value().
    Returns True if the noise weight sum is valid.
    Logs a specific warning using description and returns False if it is not.
    '''
    if not check_noise_weight_value(noise_weight_sum, description):
        return False
    return True

def get_noise_weight_default(noise_weight_config):
    '''
    Returns the default noise weight, if present in noise_weight_config.
    Otherwise, returns None.
    '''
    return noise_weight_config.get(DEFAULT_NOISE_WEIGHT_NAME, None)

def has_noise_weight_default(noise_weight_config):
    '''
    Returns True if noise_weight_config has a default noise weight.
    Otherwise, returns False.
    '''
    return get_noise_weight_default(noise_weight_config) is not None

def get_noise_weight(noise_weight_config, fingerprint):
    '''
    Returns the noise weight for fingerprint, which can be None.
    If fingerprint does not have a noise weight (or is None), return the
    default noise weight (if any).
    Otherwise, returns None.
    '''
    if fingerprint is not None and fingerprint in noise_weight_config:
        return noise_weight_config[fingerprint]
    elif has_noise_weight_default(noise_weight_config):
        return get_noise_weight_default(noise_weight_config)
    else:
        return None

def has_noise_weight(noise_weight_config, fingerprint):
    '''
    Returns True if fingerprint has a noise weight. fingerprint can be None.
    If fingerprint is None or missing, returns True if there is a default
    noise weight.
    If fingerprint does not have a noise weight, returns False.
    '''
    return get_noise_weight(noise_weight_config, fingerprint) is not None

def check_noise_weight_config(noise_weight_config, dc_threshold):
    '''
    Check that noise_weight_config is a valid noise weight configuration.
    Each noise weight must also pass check_noise_weight_value().
    Returns True if the noise weight config is valid.
    Logs a specific warning and returns False if it is not.
    '''
    if not check_dc_threshold(dc_threshold):
        return False
    # there must be noise weights for a threshold of DCs, or there must be
    # a default noise weight
    if (len(noise_weight_config) < dc_threshold and
        not has_noise_weight_default(noise_weight_config)):
        logging.warning("There must be at least as many noise weights as the threshold of data collectors, or there must be a default noise weight. Noise weights: {}, Threshold: {}."
                        .format(len(noise_weight_config), dc_threshold))
        return False
    # each noise weight must be individually valid
    for dc in noise_weight_config:
        if not check_noise_weight_value(noise_weight_config[dc]):
            return False
    # calculate the maximum possible noise weight
    noise_weight_sum = sum(noise_weight_config.values())
    # if there is a default, assume a threshold of relays might use it
    if has_noise_weight_default(noise_weight_config):
        default_weight = get_noise_weight_default(noise_weight_config)
        # adjust the sum for the extra default value
        noise_weight_sum -= default_weight
        # add a threshold of that weight
        assert dc_threshold > 0
        noise_weight_sum += dc_threshold*default_weight
    # the sum must be valid
    if not check_noise_weight_sum(noise_weight_sum):
        return False
    return True

def check_event_set_case(event_set):
    '''
    Check that event_set is a set, and each event in it has the correct case
    Returns True if all checks pass, and False if any check fails
    '''
    if not isinstance(event_set, (set, frozenset)):
        return False
    for event in event_set:
        if event != event.upper():
            return False
    return True

def check_event_set_valid(event_set):
    '''
    Check that event_set passes check_event_set_case, and also that each event
    is in the set of valid events
    Returns True if all checks pass, and False if any check fails
    '''
    if not check_event_set_case(event_set):
        return False
    for event in event_set:
        if event not in get_valid_events():
            return False
    return True

# internal
CELL_EVENT = 'PRIVCOUNT_CIRCUIT_CELL'
BYTES_EVENT = 'PRIVCOUNT_STREAM_BYTES_TRANSFERRED'
STREAM_EVENT = 'PRIVCOUNT_STREAM_ENDED'
CIRCUIT_EVENT = 'PRIVCOUNT_CIRCUIT_CLOSE'
CONNECTION_EVENT = 'PRIVCOUNT_CONNECTION_CLOSE'
HSDIR_STORE_EVENT = 'PRIVCOUNT_HSDIR_CACHE_STORE'
HSDIR_FETCH_EVENT = 'PRIVCOUNT_HSDIR_CACHE_FETCH'
VITERBI_PACKETS_EVENT = 'PRIVCOUNT_VITERBI_PACKETS'
VITERBI_STREAMS_EVENT = 'PRIVCOUNT_VITERBI_STREAMS'

# Unused events
# PrivCount never used this event, it was used by PrivEx
DNS_EVENT = 'PRIVCOUNT_DNS_RESOLVED'
# We don't use this event any more, but the Tor patch still produces it, for
# compatibility with older versions
LEGACY_CIRCUIT_EVENT = 'PRIVCOUNT_CIRCUIT_ENDED'
LEGACY_CONNECTION_EVENT = 'PRIVCOUNT_CONNECTION_ENDED'

def get_valid_events():
    '''
    Return a set containing the name of each privcount event, in uppercase
    '''
    event_set = { CELL_EVENT,
                  BYTES_EVENT,
                  STREAM_EVENT,
                  CIRCUIT_EVENT,
                  CONNECTION_EVENT,
                  HSDIR_STORE_EVENT,
                  HSDIR_FETCH_EVENT,
                  VITERBI_PACKETS_EVENT,
                  VITERBI_STREAMS_EVENT,
                  # Unused events
                  DNS_EVENT,
                  LEGACY_CIRCUIT_EVENT,
                  LEGACY_CONNECTION_EVENT,
                  }
    assert check_event_set_case(event_set)
    return event_set

# when you modify this list, update the test counters, and run:
# test/test_counter_match.sh
PRIVCOUNT_COUNTER_EVENTS = {

# these counters depend on bytes transferred event
# they are updated in _handle_circuit_cell_event_traffic_model

# these counters are for the traffic model code
# model-specific counters are added in register_dynamic_counter
# viterbi paths for packet modeling are counted on stream end events
'ExitStreamTrafficModelStreamCount' : { VITERBI_PACKETS_EVENT },
'ExitStreamTrafficModelEmissionCount' : { VITERBI_PACKETS_EVENT },
'ExitStreamTrafficModelTransitionCount' : { VITERBI_PACKETS_EVENT },
'ExitStreamTrafficModelDelayTime' : { VITERBI_PACKETS_EVENT },
'ExitStreamTrafficModelLogDelayTime' : { VITERBI_PACKETS_EVENT },
'ExitStreamTrafficModelSquaredLogDelayTime' : { VITERBI_PACKETS_EVENT },
# viterbi paths for stream modeling are counted on circuit end events
'ExitCircuitTrafficModelCircuitCount' : { VITERBI_STREAMS_EVENT },
'ExitCircuitTrafficModelEmissionCount' : { VITERBI_STREAMS_EVENT },
'ExitCircuitTrafficModelTransitionCount' : { VITERBI_STREAMS_EVENT },
'ExitCircuitTrafficModelDelayTime' : { VITERBI_STREAMS_EVENT },
'ExitCircuitTrafficModelLogDelayTime' : { VITERBI_STREAMS_EVENT },
'ExitCircuitTrafficModelSquaredLogDelayTime' : { VITERBI_STREAMS_EVENT },

'ExitStreamCount' : { STREAM_EVENT },
'ExitStreamByteCount' : { STREAM_EVENT },
'ExitStreamOutboundByteCount' : { STREAM_EVENT },
'ExitStreamInboundByteCount' : { STREAM_EVENT },
'ExitStreamByteHistogram' : { STREAM_EVENT },
'ExitStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitStreamByteRatio' : { STREAM_EVENT },
'ExitStreamLifeTime' : { STREAM_EVENT },

# Port Classification
'ExitWebStreamCount' : { STREAM_EVENT },
'ExitWebStreamByteCount' : { STREAM_EVENT },
'ExitWebStreamOutboundByteCount' : { STREAM_EVENT },
'ExitWebStreamInboundByteCount' : { STREAM_EVENT },
'ExitWebStreamByteHistogram' : { STREAM_EVENT },
'ExitWebStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitWebStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitWebStreamByteRatio' : { STREAM_EVENT },
'ExitWebStreamLifeTime' : { STREAM_EVENT },
'ExitInteractiveStreamCount' : { STREAM_EVENT },
'ExitInteractiveStreamByteCount' : { STREAM_EVENT },
'ExitInteractiveStreamOutboundByteCount' : { STREAM_EVENT },
'ExitInteractiveStreamInboundByteCount' : { STREAM_EVENT },
'ExitInteractiveStreamByteHistogram' : { STREAM_EVENT },
'ExitInteractiveStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitInteractiveStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitInteractiveStreamByteRatio' : { STREAM_EVENT },
'ExitInteractiveStreamLifeTime' : { STREAM_EVENT },
'ExitP2PStreamCount' : { STREAM_EVENT },
'ExitP2PStreamByteCount' : { STREAM_EVENT },
'ExitP2PStreamOutboundByteCount' : { STREAM_EVENT },
'ExitP2PStreamInboundByteCount' : { STREAM_EVENT },
'ExitP2PStreamByteHistogram' : { STREAM_EVENT },
'ExitP2PStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitP2PStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitP2PStreamByteRatio' : { STREAM_EVENT },
'ExitP2PStreamLifeTime' : { STREAM_EVENT },
'ExitOtherPortStreamCount' : { STREAM_EVENT },
'ExitOtherPortStreamByteCount' : { STREAM_EVENT },
'ExitOtherPortStreamOutboundByteCount' : { STREAM_EVENT },
'ExitOtherPortStreamInboundByteCount' : { STREAM_EVENT },
'ExitOtherPortStreamByteHistogram' : { STREAM_EVENT },
'ExitOtherPortStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitOtherPortStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitOtherPortStreamByteRatio' : { STREAM_EVENT },
'ExitOtherPortStreamLifeTime' : { STREAM_EVENT },

# Is this stream *not* on port 80 or 443?
# Includes Interactive, P2P, and Other
'ExitNonWebStreamCount' : { STREAM_EVENT },
'ExitNonWebStreamByteCount' : { STREAM_EVENT },
'ExitNonWebStreamOutboundByteCount' : { STREAM_EVENT },
'ExitNonWebStreamInboundByteCount' : { STREAM_EVENT },
'ExitNonWebStreamByteHistogram' : { STREAM_EVENT },
'ExitNonWebStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitNonWebStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitNonWebStreamByteRatio' : { STREAM_EVENT },
'ExitNonWebStreamLifeTime' : { STREAM_EVENT },

# IP version after DNS resolution
'ExitIPv4StreamCount' : { STREAM_EVENT },
'ExitIPv4StreamByteCount' : { STREAM_EVENT },
'ExitIPv4StreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv4StreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv4StreamByteHistogram' : { STREAM_EVENT },
'ExitIPv4StreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4StreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4StreamByteRatio' : { STREAM_EVENT },
'ExitIPv4StreamLifeTime' : { STREAM_EVENT },

'ExitIPv6StreamCount' : { STREAM_EVENT },
'ExitIPv6StreamByteCount' : { STREAM_EVENT },
'ExitIPv6StreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv6StreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv6StreamByteHistogram' : { STREAM_EVENT },
'ExitIPv6StreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6StreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6StreamByteRatio' : { STREAM_EVENT },
'ExitIPv6StreamLifeTime' : { STREAM_EVENT },

# IP version or hostname before DNS resolution
'ExitIPv4LiteralStreamCount' : { STREAM_EVENT },
'ExitIPv4LiteralStreamByteCount' : { STREAM_EVENT },
'ExitIPv4LiteralStreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv4LiteralStreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv4LiteralStreamByteHistogram' : { STREAM_EVENT },
'ExitIPv4LiteralStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4LiteralStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4LiteralStreamByteRatio' : { STREAM_EVENT },
'ExitIPv4LiteralStreamLifeTime' : { STREAM_EVENT },

'ExitIPv6LiteralStreamCount' : { STREAM_EVENT },
'ExitIPv6LiteralStreamByteCount' : { STREAM_EVENT },
'ExitIPv6LiteralStreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv6LiteralStreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv6LiteralStreamByteHistogram' : { STREAM_EVENT },
'ExitIPv6LiteralStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6LiteralStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6LiteralStreamByteRatio' : { STREAM_EVENT },
'ExitIPv6LiteralStreamLifeTime' : { STREAM_EVENT },

'ExitHostnameStreamCount' : { STREAM_EVENT },
'ExitHostnameStreamByteCount' : { STREAM_EVENT },
'ExitHostnameStreamOutboundByteCount' : { STREAM_EVENT },
'ExitHostnameStreamInboundByteCount' : { STREAM_EVENT },
'ExitHostnameStreamByteHistogram' : { STREAM_EVENT },
'ExitHostnameStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameStreamByteRatio' : { STREAM_EVENT },
'ExitHostnameStreamLifeTime' : { STREAM_EVENT },

# Hostnames on Web and Non-Web streams
'ExitHostnameWebStreamCount' : { STREAM_EVENT },
'ExitHostnameWebStreamByteCount' : { STREAM_EVENT },
'ExitHostnameWebStreamOutboundByteCount' : { STREAM_EVENT },
'ExitHostnameWebStreamInboundByteCount' : { STREAM_EVENT },
'ExitHostnameWebStreamByteHistogram' : { STREAM_EVENT },
'ExitHostnameWebStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameWebStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameWebStreamByteRatio' : { STREAM_EVENT },
'ExitHostnameWebStreamLifeTime' : { STREAM_EVENT },

'ExitHostnameNonWebStreamCount' : { STREAM_EVENT },
'ExitHostnameNonWebStreamByteCount' : { STREAM_EVENT },
'ExitHostnameNonWebStreamOutboundByteCount' : { STREAM_EVENT },
'ExitHostnameNonWebStreamInboundByteCount' : { STREAM_EVENT },
'ExitHostnameNonWebStreamByteHistogram' : { STREAM_EVENT },
'ExitHostnameNonWebStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameNonWebStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameNonWebStreamByteRatio' : { STREAM_EVENT },
'ExitHostnameNonWebStreamLifeTime' : { STREAM_EVENT },

# Position of stream on circuit
# These also use CIRCUIT_EVENT, because that avoids collisions between old and
# new streams with the same circuit id. See #451.
'ExitInitialStreamCount' : { STREAM_EVENT },
'ExitInitialStreamByteCount' : { STREAM_EVENT },
'ExitInitialStreamOutboundByteCount' : { STREAM_EVENT },
'ExitInitialStreamInboundByteCount' : { STREAM_EVENT },
'ExitInitialStreamByteHistogram' : { STREAM_EVENT },
'ExitInitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitInitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitInitialStreamByteRatio' : { STREAM_EVENT },
'ExitInitialStreamLifeTime' : { STREAM_EVENT },

'ExitSubsequentStreamCount' : { STREAM_EVENT },
'ExitSubsequentStreamByteCount' : { STREAM_EVENT },
'ExitSubsequentStreamOutboundByteCount' : { STREAM_EVENT },
'ExitSubsequentStreamInboundByteCount' : { STREAM_EVENT },
'ExitSubsequentStreamByteHistogram' : { STREAM_EVENT },
'ExitSubsequentStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitSubsequentStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitSubsequentStreamByteRatio' : { STREAM_EVENT },
'ExitSubsequentStreamLifeTime' : { STREAM_EVENT },

# IP version after DNS resolution and position
'ExitIPv4InitialStreamCount' : { STREAM_EVENT },
'ExitIPv4InitialStreamByteCount' : { STREAM_EVENT },
'ExitIPv4InitialStreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv4InitialStreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv4InitialStreamByteHistogram' : { STREAM_EVENT },
'ExitIPv4InitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4InitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4InitialStreamByteRatio' : { STREAM_EVENT },
'ExitIPv4InitialStreamLifeTime' : { STREAM_EVENT },

'ExitIPv6InitialStreamCount' : { STREAM_EVENT },
'ExitIPv6InitialStreamByteCount' : { STREAM_EVENT },
'ExitIPv6InitialStreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv6InitialStreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv6InitialStreamByteHistogram' : { STREAM_EVENT },
'ExitIPv6InitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6InitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6InitialStreamByteRatio' : { STREAM_EVENT },
'ExitIPv6InitialStreamLifeTime' : { STREAM_EVENT },

# IP version or hostname before DNS resolution and position
'ExitIPv4LiteralInitialStreamCount' : { STREAM_EVENT },
'ExitIPv4LiteralInitialStreamByteCount' : { STREAM_EVENT },
'ExitIPv4LiteralInitialStreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv4LiteralInitialStreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv4LiteralInitialStreamByteHistogram' : { STREAM_EVENT },
'ExitIPv4LiteralInitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4LiteralInitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4LiteralInitialStreamByteRatio' : { STREAM_EVENT },
'ExitIPv4LiteralInitialStreamLifeTime' : { STREAM_EVENT },

'ExitIPv6LiteralInitialStreamCount' : { STREAM_EVENT },
'ExitIPv6LiteralInitialStreamByteCount' : { STREAM_EVENT },
'ExitIPv6LiteralInitialStreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv6LiteralInitialStreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv6LiteralInitialStreamByteHistogram' : { STREAM_EVENT },
'ExitIPv6LiteralInitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6LiteralInitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6LiteralInitialStreamByteRatio' : { STREAM_EVENT },
'ExitIPv6LiteralInitialStreamLifeTime' : { STREAM_EVENT },

'ExitHostnameInitialStreamCount' : { STREAM_EVENT },
'ExitHostnameInitialStreamByteCount' : { STREAM_EVENT },
'ExitHostnameInitialStreamOutboundByteCount' : { STREAM_EVENT },
'ExitHostnameInitialStreamInboundByteCount' : { STREAM_EVENT },
'ExitHostnameInitialStreamByteHistogram' : { STREAM_EVENT },
'ExitHostnameInitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameInitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameInitialStreamByteRatio' : { STREAM_EVENT },
'ExitHostnameInitialStreamLifeTime' : { STREAM_EVENT },

# The base counts for the ExitDomain*Web*Stream* counters
'ExitHostnameWebInitialStreamCount' : { STREAM_EVENT },
'ExitHostnameWebInitialStreamByteCount' : { STREAM_EVENT },
'ExitHostnameWebInitialStreamOutboundByteCount' : { STREAM_EVENT },
'ExitHostnameWebInitialStreamInboundByteCount' : { STREAM_EVENT },
'ExitHostnameWebInitialStreamByteHistogram' : { STREAM_EVENT },
'ExitHostnameWebInitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameWebInitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameWebInitialStreamByteRatio' : { STREAM_EVENT },
'ExitHostnameWebInitialStreamLifeTime' : { STREAM_EVENT },

'ExitHostnameWebSubsequentStreamCount' : { STREAM_EVENT },
'ExitHostnameWebSubsequentStreamByteCount' : { STREAM_EVENT },
'ExitHostnameWebSubsequentStreamOutboundByteCount' : { STREAM_EVENT },
'ExitHostnameWebSubsequentStreamInboundByteCount' : { STREAM_EVENT },
'ExitHostnameWebSubsequentStreamByteHistogram' : { STREAM_EVENT },
'ExitHostnameWebSubsequentStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameWebSubsequentStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameWebSubsequentStreamByteRatio' : { STREAM_EVENT },
'ExitHostnameWebSubsequentStreamLifeTime' : { STREAM_EVENT },

# The non-web equivalents of ExitHostnameWebInitial/SubsequentStream*
'ExitHostnameNonWebInitialStreamCount' : { STREAM_EVENT },
'ExitHostnameNonWebInitialStreamByteCount' : { STREAM_EVENT },
'ExitHostnameNonWebInitialStreamOutboundByteCount' : { STREAM_EVENT },
'ExitHostnameNonWebInitialStreamInboundByteCount' : { STREAM_EVENT },
'ExitHostnameNonWebInitialStreamByteHistogram' : { STREAM_EVENT },
'ExitHostnameNonWebInitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameNonWebInitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameNonWebInitialStreamByteRatio' : { STREAM_EVENT },
'ExitHostnameNonWebInitialStreamLifeTime' : { STREAM_EVENT },

'ExitHostnameNonWebSubsequentStreamCount' : { STREAM_EVENT },
'ExitHostnameNonWebSubsequentStreamByteCount' : { STREAM_EVENT },
'ExitHostnameNonWebSubsequentStreamOutboundByteCount' : { STREAM_EVENT },
'ExitHostnameNonWebSubsequentStreamInboundByteCount' : { STREAM_EVENT },
'ExitHostnameNonWebSubsequentStreamByteHistogram' : { STREAM_EVENT },
'ExitHostnameNonWebSubsequentStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameNonWebSubsequentStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameNonWebSubsequentStreamByteRatio' : { STREAM_EVENT },
'ExitHostnameNonWebSubsequentStreamLifeTime' : { STREAM_EVENT },

# IP version after DNS resolution and position
'ExitIPv4SubsequentStreamCount' : { STREAM_EVENT },
'ExitIPv4SubsequentStreamByteCount' : { STREAM_EVENT },
'ExitIPv4SubsequentStreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv4SubsequentStreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv4SubsequentStreamByteHistogram' : { STREAM_EVENT },
'ExitIPv4SubsequentStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4SubsequentStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4SubsequentStreamByteRatio' : { STREAM_EVENT },
'ExitIPv4SubsequentStreamLifeTime' : { STREAM_EVENT },

'ExitIPv6SubsequentStreamCount' : { STREAM_EVENT },
'ExitIPv6SubsequentStreamByteCount' : { STREAM_EVENT },
'ExitIPv6SubsequentStreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv6SubsequentStreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv6SubsequentStreamByteHistogram' : { STREAM_EVENT },
'ExitIPv6SubsequentStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6SubsequentStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6SubsequentStreamByteRatio' : { STREAM_EVENT },
'ExitIPv6SubsequentStreamLifeTime' : { STREAM_EVENT },

# IP version or hostname before DNS resolution and position
'ExitIPv4LiteralSubsequentStreamCount' : { STREAM_EVENT },
'ExitIPv4LiteralSubsequentStreamByteCount' : { STREAM_EVENT },
'ExitIPv4LiteralSubsequentStreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv4LiteralSubsequentStreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv4LiteralSubsequentStreamByteHistogram' : { STREAM_EVENT },
'ExitIPv4LiteralSubsequentStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4LiteralSubsequentStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv4LiteralSubsequentStreamByteRatio' : { STREAM_EVENT },
'ExitIPv4LiteralSubsequentStreamLifeTime' : { STREAM_EVENT },

'ExitIPv6LiteralSubsequentStreamCount' : { STREAM_EVENT },
'ExitIPv6LiteralSubsequentStreamByteCount' : { STREAM_EVENT },
'ExitIPv6LiteralSubsequentStreamOutboundByteCount' : { STREAM_EVENT },
'ExitIPv6LiteralSubsequentStreamInboundByteCount' : { STREAM_EVENT },
'ExitIPv6LiteralSubsequentStreamByteHistogram' : { STREAM_EVENT },
'ExitIPv6LiteralSubsequentStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6LiteralSubsequentStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitIPv6LiteralSubsequentStreamByteRatio' : { STREAM_EVENT },
'ExitIPv6LiteralSubsequentStreamLifeTime' : { STREAM_EVENT },

'ExitHostnameSubsequentStreamCount' : { STREAM_EVENT },
'ExitHostnameSubsequentStreamByteCount' : { STREAM_EVENT },
'ExitHostnameSubsequentStreamOutboundByteCount' : { STREAM_EVENT },
'ExitHostnameSubsequentStreamInboundByteCount' : { STREAM_EVENT },
'ExitHostnameSubsequentStreamByteHistogram' : { STREAM_EVENT },
'ExitHostnameSubsequentStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameSubsequentStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitHostnameSubsequentStreamByteRatio' : { STREAM_EVENT },
'ExitHostnameSubsequentStreamLifeTime' : { STREAM_EVENT },

# The first domain list is used for the ExitDomain*MatchWebInitialStream Ratio, LifeTime, and Histogram counters
# Their ExitDomainNo*MatchWebInitialStream* equivalents are used when there is no match in the first list

# Does the initial domain on the circuit match any domain in the first list?
'ExitDomainExactMatchWebInitialStreamByteHistogram' : { STREAM_EVENT },
'ExitDomainExactMatchWebInitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitDomainExactMatchWebInitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitDomainExactMatchWebInitialStreamByteRatio' : { STREAM_EVENT },
'ExitDomainExactMatchWebInitialStreamLifeTime' : { STREAM_EVENT },

'ExitDomainNoExactMatchWebInitialStreamByteHistogram' : { STREAM_EVENT },
'ExitDomainNoExactMatchWebInitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitDomainNoExactMatchWebInitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitDomainNoExactMatchWebInitialStreamByteRatio' : { STREAM_EVENT },
'ExitDomainNoExactMatchWebInitialStreamLifeTime' : { STREAM_EVENT },

# Does the initial domain on the circuit have any domain in the first list as a suffix?
'ExitDomainSuffixMatchWebInitialStreamByteHistogram' : { STREAM_EVENT },
'ExitDomainSuffixMatchWebInitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitDomainSuffixMatchWebInitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitDomainSuffixMatchWebInitialStreamByteRatio' : { STREAM_EVENT },
'ExitDomainSuffixMatchWebInitialStreamLifeTime' : { STREAM_EVENT },

'ExitDomainNoSuffixMatchWebInitialStreamByteHistogram' : { STREAM_EVENT },
'ExitDomainNoSuffixMatchWebInitialStreamOutboundByteHistogram' : { STREAM_EVENT },
'ExitDomainNoSuffixMatchWebInitialStreamInboundByteHistogram' : { STREAM_EVENT },
'ExitDomainNoSuffixMatchWebInitialStreamByteRatio' : { STREAM_EVENT },
'ExitDomainNoSuffixMatchWebInitialStreamLifeTime' : { STREAM_EVENT },

# The number of bins in the ExitDomain*MatchWebInitialStream*CountList counters is
# determined at runtime, based on the number of configured domain lists
# Each domain list gets a bin in each counter, and there is a final bin
# for "no match in any list" (multiple lists may match: all matching bins
# will be incremented). Since there is an unmatched bin, there are no
# ExitDomainNo*MatchWebInitialStream*CountList counters.

# Does the initial domain on the circuit match any domain in the list for each bin? Or is it unmatched by all the lists?
'ExitDomainExactMatchWebInitialStreamCountList' : { STREAM_EVENT },
'ExitDomainExactMatchWebInitialStreamByteCountList' : { STREAM_EVENT },
'ExitDomainExactMatchWebInitialStreamOutboundByteCountList' : { STREAM_EVENT },
'ExitDomainExactMatchWebInitialStreamInboundByteCountList' : { STREAM_EVENT },

# Does the initial domain on the circuit have any domain in the list for each bin as a suffix? Or is it unmatched by all the lists?
'ExitDomainSuffixMatchWebInitialStreamCountList' : { STREAM_EVENT },
'ExitDomainSuffixMatchWebInitialStreamByteCountList' : { STREAM_EVENT },
'ExitDomainSuffixMatchWebInitialStreamOutboundByteCountList' : { STREAM_EVENT },
'ExitDomainSuffixMatchWebInitialStreamInboundByteCountList' : { STREAM_EVENT },

# these counters depend on circuit end
# they are updated in _handle_circuit_close_event

# Non-HS Circuit Positions

# Custom circuit counters
'ExitAndRend2ClientCircuitCount' : { CIRCUIT_EVENT },
'ExitAndRend2ServiceCircuitCount' : { CIRCUIT_EVENT },

# Circuit Counts
# Inbound cells travel towards the origin
# Outbound cells travel towards the end

'OriginCircuitCount' : { CIRCUIT_EVENT },
'OriginCircuitInboundCellCount' : { CIRCUIT_EVENT },
'OriginCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'OriginCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'OriginCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'OriginCircuitLifeTime' : { CIRCUIT_EVENT },
'OriginFailureCircuitCount' : { CIRCUIT_EVENT },
'OriginFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'OriginFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'OriginFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'OriginFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'OriginFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'OriginSuccessCircuitCount' : { CIRCUIT_EVENT },
'OriginSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'OriginSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'OriginSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'OriginSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'OriginSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'OriginActiveCircuitCount' : { CIRCUIT_EVENT },
'OriginActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'OriginActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'OriginActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'OriginActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'OriginActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'OriginActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'OriginActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'OriginActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'OriginActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'OriginActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'OriginActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'OriginActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'OriginActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'OriginActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'OriginActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'OriginActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'OriginActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'OriginActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'OriginActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'OriginActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'OriginInactiveCircuitCount' : { CIRCUIT_EVENT },
'OriginInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'OriginInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'OriginInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'OriginInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'OriginInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'OriginInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'OriginInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'OriginInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'OriginInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'OriginInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'OriginInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'OriginInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'OriginInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'OriginInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'OriginInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'OriginInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'OriginInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'EntryCircuitCount' : { CIRCUIT_EVENT },
'EntryCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EntryCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EntryCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EntryCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EntryCircuitLifeTime' : { CIRCUIT_EVENT },
'EntryFailureCircuitCount' : { CIRCUIT_EVENT },
'EntryFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EntryFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EntryFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EntryFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EntryFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'EntrySuccessCircuitCount' : { CIRCUIT_EVENT },
'EntrySuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EntrySuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EntrySuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EntrySuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EntrySuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'EntryActiveCircuitCount' : { CIRCUIT_EVENT },
'EntryActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EntryActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EntryActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EntryActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EntryActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'EntryActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'EntryActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'EntryActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EntryActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EntryActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EntryActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EntryActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'EntryActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'EntryActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'EntryActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EntryActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EntryActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EntryActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EntryActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'EntryActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'EntryInactiveCircuitCount' : { CIRCUIT_EVENT },
'EntryInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EntryInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EntryInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EntryInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EntryInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'EntryInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'EntryInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EntryInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EntryInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EntryInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EntryInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'EntryInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'EntryInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EntryInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EntryInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EntryInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EntryInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'MidCircuitCount' : { CIRCUIT_EVENT },
'MidCircuitInboundCellCount' : { CIRCUIT_EVENT },
'MidCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'MidCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'MidCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'MidCircuitLifeTime' : { CIRCUIT_EVENT },
'MidFailureCircuitCount' : { CIRCUIT_EVENT },
'MidFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'MidFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'MidFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'MidFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'MidFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'MidSuccessCircuitCount' : { CIRCUIT_EVENT },
'MidSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'MidSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'MidSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'MidSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'MidSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'MidActiveCircuitCount' : { CIRCUIT_EVENT },
'MidActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'MidActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'MidActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'MidActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'MidActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'MidActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'MidActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'MidActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'MidActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'MidActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'MidActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'MidActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'MidActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'MidActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'MidActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'MidActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'MidActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'MidActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'MidActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'MidActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'MidInactiveCircuitCount' : { CIRCUIT_EVENT },
'MidInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'MidInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'MidInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'MidInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'MidInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'MidInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'MidInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'MidInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'MidInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'MidInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'MidInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'MidInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'MidInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'MidInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'MidInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'MidInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'MidInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'EndCircuitCount' : { CIRCUIT_EVENT },
'EndCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EndCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EndCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EndCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EndCircuitLifeTime' : { CIRCUIT_EVENT },
'EndFailureCircuitCount' : { CIRCUIT_EVENT },
'EndFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EndFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EndFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EndFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EndFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'EndSuccessCircuitCount' : { CIRCUIT_EVENT },
'EndSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EndSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EndSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EndSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EndSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'EndActiveCircuitCount' : { CIRCUIT_EVENT },
'EndActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EndActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EndActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EndActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EndActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'EndActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'EndActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'EndActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EndActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EndActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EndActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EndActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'EndActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'EndActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'EndActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EndActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EndActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EndActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EndActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'EndActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'EndInactiveCircuitCount' : { CIRCUIT_EVENT },
'EndInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EndInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EndInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EndInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EndInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'EndInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'EndInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EndInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EndInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EndInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EndInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'EndInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'EndInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'EndInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'EndInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'EndInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'EndInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'SingleHopCircuitCount' : { CIRCUIT_EVENT },
'SingleHopCircuitInboundCellCount' : { CIRCUIT_EVENT },
'SingleHopCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'SingleHopCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopCircuitLifeTime' : { CIRCUIT_EVENT },
'SingleHopFailureCircuitCount' : { CIRCUIT_EVENT },
'SingleHopFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'SingleHopFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'SingleHopFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'SingleHopSuccessCircuitCount' : { CIRCUIT_EVENT },
'SingleHopSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'SingleHopSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'SingleHopSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'SingleHopActiveCircuitCount' : { CIRCUIT_EVENT },
'SingleHopActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'SingleHopActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'SingleHopActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'SingleHopActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'SingleHopActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'SingleHopActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'SingleHopActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'SingleHopActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'SingleHopActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'SingleHopActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'SingleHopActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'SingleHopActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'SingleHopActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'SingleHopActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'SingleHopInactiveCircuitCount' : { CIRCUIT_EVENT },
'SingleHopInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'SingleHopInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'SingleHopInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'SingleHopInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'SingleHopInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'SingleHopInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'SingleHopInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'SingleHopInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'SingleHopInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'SingleHopInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'SingleHopInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'SingleHopInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

# We can't distinguish inactive Exit, Dir, and HSDir: we learn if an End
# is Exit, Dir, or HSDir after a stream opens. And all circuits with open
# streams are considered active.
# Use the End position to count inactive circuits.

'ExitCircuitCount' : { CIRCUIT_EVENT },
'ExitCircuitInboundCellCount' : { CIRCUIT_EVENT },
'ExitCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'ExitCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'ExitCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'ExitCircuitCellRatio' : { CIRCUIT_EVENT },
'ExitCircuitLifeTime' : { CIRCUIT_EVENT },
'ExitFailureCircuitCount' : { CIRCUIT_EVENT },
'ExitFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'ExitFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'ExitFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'ExitFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'ExitFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'ExitFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'ExitSuccessCircuitCount' : { CIRCUIT_EVENT },
'ExitSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'ExitSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'ExitSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'ExitSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'ExitSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'ExitSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'DirCircuitCount' : { CIRCUIT_EVENT },
'DirCircuitInboundCellCount' : { CIRCUIT_EVENT },
'DirCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'DirCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'DirCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'DirCircuitCellRatio' : { CIRCUIT_EVENT },
'DirCircuitLifeTime' : { CIRCUIT_EVENT },
'DirFailureCircuitCount' : { CIRCUIT_EVENT },
'DirFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'DirFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'DirFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'DirFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'DirFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'DirFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'DirSuccessCircuitCount' : { CIRCUIT_EVENT },
'DirSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'DirSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'DirSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'DirSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'DirSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'DirSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

# HSDir circuits
# You probably want the HSDir*Store/Fetch* events instead of these events
'HSDirCircuitCount' : { CIRCUIT_EVENT },
'HSDirCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDirFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDirSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDirClientCircuitCount' : { CIRCUIT_EVENT },
'HSDirClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirClientCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirClientCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirClientFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDirClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirClientFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDirClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirClientSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDirServiceCircuitCount' : { CIRCUIT_EVENT },
'HSDirServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirServiceCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDirServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirServiceFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDirServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirServiceSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDirTor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'HSDirTor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirTor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirTor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirTor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirTor2WebClientCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirTor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirTor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDirTor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirTor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirTor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirTor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirTor2WebClientFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirTor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirTor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDirTor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirTor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirTor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirTor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirTor2WebClientSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirTor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDirSingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDirMultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'HSDirMultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopClientCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirMultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirMultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDirMultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopClientFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirMultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirMultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDirMultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopClientSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirMultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDirMultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'HSDir2CircuitCount' : { CIRCUIT_EVENT },
'HSDir2CircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2CircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2CircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2CircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2CircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2CircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2FailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir2FailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2FailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2FailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2FailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2FailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2FailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2SuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir2SuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2SuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2SuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2SuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2SuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2SuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir2ClientCircuitCount' : { CIRCUIT_EVENT },
'HSDir2ClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ClientCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2ClientCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2ClientFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir2ClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ClientFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2ClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2ClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir2ClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ClientSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2ClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir2ServiceCircuitCount' : { CIRCUIT_EVENT },
'HSDir2ServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ServiceCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2ServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2ServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir2ServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ServiceFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2ServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2ServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir2ServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2ServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2ServiceSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2ServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir2Tor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir2SingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir2MultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir2MultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'HSDir3CircuitCount' : { CIRCUIT_EVENT },
'HSDir3CircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3CircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3CircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3CircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3CircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3CircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3FailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir3FailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3FailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3FailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3FailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3FailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3FailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3SuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir3SuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3SuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3SuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3SuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3SuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3SuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir3ClientCircuitCount' : { CIRCUIT_EVENT },
'HSDir3ClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ClientCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3ClientCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3ClientFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir3ClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ClientFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3ClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3ClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir3ClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ClientSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3ClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir3ServiceCircuitCount' : { CIRCUIT_EVENT },
'HSDir3ServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ServiceCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3ServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3ServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir3ServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ServiceFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3ServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3ServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir3ServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3ServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3ServiceSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3ServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir3Tor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir3SingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir3MultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'HSDir3MultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

# Intro and Rend Circuits

'IntroCircuitCount' : { CIRCUIT_EVENT },
'IntroCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroClientCircuitCount' : { CIRCUIT_EVENT },
'IntroClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroClientFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroClientActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroServiceCircuitCount' : { CIRCUIT_EVENT },
'IntroServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroTor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroTor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroTor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroTor2WebClientActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroTor2WebClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroSingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroSingleOnionServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroSingleOnionServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroMultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroMultiHopClientActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroMultiHopClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroMultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroMultiHopServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroMultiHopServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },



'IntroUCircuitCount' : { CIRCUIT_EVENT },
'IntroUCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroUClientCircuitCount' : { CIRCUIT_EVENT },
'IntroUClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUClientFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUClientActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroUServiceCircuitCount' : { CIRCUIT_EVENT },
'IntroUServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroUTor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUTor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUTor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUTor2WebClientActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUTor2WebClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroUSingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUSingleOnionServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUSingleOnionServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroUMultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUMultiHopClientActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUMultiHopClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'IntroUMultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUMultiHopServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'IntroUMultiHopServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },



'Intro2CircuitCount' : { CIRCUIT_EVENT },
'Intro2CircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2CircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2CircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2CircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2CircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2FailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2FailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2FailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2FailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2FailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2FailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2SuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2SuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2SuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2SuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2ActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2ActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2ActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2ActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2ActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2ActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2ActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2InactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2InactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2InactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2InactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2InactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2InactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2InactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2InactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2InactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2InactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2InactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2InactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2InactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2InactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2InactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2InactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2InactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2InactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro2ClientCircuitCount' : { CIRCUIT_EVENT },
'Intro2ClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2ClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2ClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2ClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2ClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2ClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2ClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2ClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2ClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2ClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2ClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2ClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2ClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2ClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro2ServiceCircuitCount' : { CIRCUIT_EVENT },
'Intro2ServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2ServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2ServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2ServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2ServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2ServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2ServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2ServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2ServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2ServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2ServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro2Tor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2Tor2WebClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2Tor2WebClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro2SingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2SingleOnionServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2SingleOnionServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro2MultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2MultiHopClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2MultiHopClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro2MultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2MultiHopServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro2MultiHopServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },



'Intro3CircuitCount' : { CIRCUIT_EVENT },
'Intro3CircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3CircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3CircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3CircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3CircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3FailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3FailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3FailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3FailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3FailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3FailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3SuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3SuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3SuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3SuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3ActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3ActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3ActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3ActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3ActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3ActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3ActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3InactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3InactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3InactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3InactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3InactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3InactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3InactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3InactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3InactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3InactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3InactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3InactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3InactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3InactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3InactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3InactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3InactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3InactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro3ClientCircuitCount' : { CIRCUIT_EVENT },
'Intro3ClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3ClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3ClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3ClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3ClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3ClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3ClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3ClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3ClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3ClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3ClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3ClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3ClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3ClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro3ServiceCircuitCount' : { CIRCUIT_EVENT },
'Intro3ServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3ServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3ServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3ServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3ServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3ServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3ServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3ServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3ServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3ServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3ServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro3Tor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3Tor2WebClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3Tor2WebClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro3SingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3SingleOnionServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3SingleOnionServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro3MultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3MultiHopClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3MultiHopClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Intro3MultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3MultiHopServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Intro3MultiHopServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },



'RendCircuitCount' : { CIRCUIT_EVENT },
'RendCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendCircuitLifeTime' : { CIRCUIT_EVENT },
'RendFailureCircuitCount' : { CIRCUIT_EVENT },
'RendFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendActiveCircuitCount' : { CIRCUIT_EVENT },
'RendActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendClientCircuitCount' : { CIRCUIT_EVENT },
'RendClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientCircuitLifeTime' : { CIRCUIT_EVENT },
'RendClientFailureCircuitCount' : { CIRCUIT_EVENT },
'RendClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendClientActiveCircuitCount' : { CIRCUIT_EVENT },
'RendClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendServiceCircuitCount' : { CIRCUIT_EVENT },
'RendServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'RendServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'RendServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'RendServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendTor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'RendTor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'RendTor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'RendTor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendTor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendTor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendTor2WebClientActiveCircuitCount' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendTor2WebClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendSingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'RendSingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendSingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendSingleOnionServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendSingleOnionServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendMultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendMultiHopClientActiveCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendMultiHopClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendMultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendMultiHopServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendMultiHopServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },



'RendUCircuitCount' : { CIRCUIT_EVENT },
'RendUCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUActiveCircuitCount' : { CIRCUIT_EVENT },
'RendUActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendUInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendUClientCircuitCount' : { CIRCUIT_EVENT },
'RendUClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUClientFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUClientActiveCircuitCount' : { CIRCUIT_EVENT },
'RendUClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendUClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendUServiceCircuitCount' : { CIRCUIT_EVENT },
'RendUServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'RendUServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendUServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendUTor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUTor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUTor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUTor2WebClientActiveCircuitCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUTor2WebClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendUSingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUSingleOnionServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUSingleOnionServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendUMultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUMultiHopClientActiveCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUMultiHopClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'RendUMultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUMultiHopServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'RendUMultiHopServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },



'Rend2CircuitCount' : { CIRCUIT_EVENT },
'Rend2CircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2CircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2CircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2CircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2CircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2FailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2FailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2FailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2FailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2FailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2FailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2SuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2SuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2SuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2SuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2ActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2ActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2ActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2ActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2ActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2ActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2ActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2InactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2InactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2InactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2InactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2InactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2InactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2InactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2InactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2InactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2InactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2InactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2InactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2InactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2InactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2InactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2InactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2InactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2InactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend2ClientCircuitCount' : { CIRCUIT_EVENT },
'Rend2ClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2ClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2ClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2ClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2ClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2ClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2ClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2ClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2ClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2ClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2ClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2ClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2ClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2ClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend2ServiceCircuitCount' : { CIRCUIT_EVENT },
'Rend2ServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2ServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2ServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2ServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2ServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2ServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2ServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2ServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2ServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2ServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2ServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend2Tor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2Tor2WebClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2Tor2WebClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend2SingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2SingleOnionServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2SingleOnionServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend2MultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2MultiHopClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2MultiHopClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend2MultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2MultiHopServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend2MultiHopServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },



'Rend3CircuitCount' : { CIRCUIT_EVENT },
'Rend3CircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3CircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3CircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3CircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3CircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3FailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3FailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3FailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3FailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3FailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3FailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3SuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3SuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3SuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3SuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3ActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3ActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3ActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3ActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3ActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3ActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3ActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3InactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3InactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3InactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3InactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3InactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3InactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3InactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3InactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3InactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3InactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3InactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3InactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3InactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3InactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3InactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3InactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3InactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3InactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend3ClientCircuitCount' : { CIRCUIT_EVENT },
'Rend3ClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3ClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3ClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3ClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3ClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3ClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3ClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3ClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3ClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3ClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3ClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3ClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3ClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3ClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend3ServiceCircuitCount' : { CIRCUIT_EVENT },
'Rend3ServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3ServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3ServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3ServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3ServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3ServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3ServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3ServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3ServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3ServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3ServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend3Tor2WebClientCircuitCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3Tor2WebClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3Tor2WebClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend3SingleOnionServiceCircuitCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3SingleOnionServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3SingleOnionServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend3MultiHopClientCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopClientFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopClientSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3MultiHopClientActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3MultiHopClientInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },


'Rend3MultiHopServiceCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3MultiHopServiceActiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveFailureCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveSuccessCircuitCellRatio' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

'Rend3MultiHopServiceInactiveCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveFailureCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveFailureCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveFailureCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveFailureCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveFailureCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveFailureCircuitLifeTime' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveSuccessCircuitCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveSuccessCircuitInboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveSuccessCircuitInboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveSuccessCircuitOutboundCellCount' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveSuccessCircuitOutboundCellHistogram' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveSuccessCircuitLifeTime' : { CIRCUIT_EVENT },

# circuit failure reason count lists

'OriginFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'OriginActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'OriginInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'EntryFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'EntryActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'EntryInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'MidFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'MidActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'MidInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'EndFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'EndActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'EndInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'SingleHopFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'SingleHopActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'SingleHopInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'ExitFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'DirFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'HSDirFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDirClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDirServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDirTor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDirSingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDirMultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDirMultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'HSDir2FailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir2ClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir2ServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir2Tor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir2SingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir2MultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir2MultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'HSDir3FailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir3ClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir3ServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir3Tor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir3SingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir3MultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'HSDir3MultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },


'IntroFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroTor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroTor2WebClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroTor2WebClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroSingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroSingleOnionServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroMultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroMultiHopClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroMultiHopClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroMultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroMultiHopServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroMultiHopServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },


'IntroUFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroUClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroUServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroUTor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUTor2WebClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUTor2WebClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroUSingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUSingleOnionServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroUMultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUMultiHopClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUMultiHopClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'IntroUMultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'IntroUMultiHopServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },


'Intro2FailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2ActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2InactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro2ClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2ClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2ClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro2ServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2ServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2ServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro2Tor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2Tor2WebClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro2SingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2SingleOnionServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro2MultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2MultiHopClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2MultiHopClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro2MultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro2MultiHopServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },


'Intro3FailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3ActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3InactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro3ClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3ClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3ClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro3ServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3ServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3ServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro3Tor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3Tor2WebClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro3SingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3SingleOnionServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro3MultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3MultiHopClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3MultiHopClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Intro3MultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Intro3MultiHopServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },


'RendFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendTor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendTor2WebClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendTor2WebClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendSingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendSingleOnionServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendSingleOnionServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendMultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendMultiHopClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendMultiHopClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendMultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendMultiHopServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendMultiHopServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },


'RendUFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendUClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendUServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendUTor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUTor2WebClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUTor2WebClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendUSingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUSingleOnionServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendUMultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUMultiHopClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUMultiHopClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'RendUMultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUMultiHopServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'RendUMultiHopServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },


'Rend2FailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2ActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2InactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend2ClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2ClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2ClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend2ServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2ServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2ServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend2Tor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2Tor2WebClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend2SingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2SingleOnionServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend2MultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2MultiHopClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2MultiHopClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend2MultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend2MultiHopServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },


'Rend3FailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3ActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3InactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend3ClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3ClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3ClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend3ServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3ServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3ServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend3Tor2WebClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3Tor2WebClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend3SingleOnionServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3SingleOnionServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend3MultiHopClientFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3MultiHopClientActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3MultiHopClientInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

'Rend3MultiHopServiceFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceActiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },
'Rend3MultiHopServiceInactiveFailureCircuitReasonCountList' : { CIRCUIT_EVENT },

# these counters depend on circuit end
# they are updated in _do_rotate,
# and use data updated in _handle_legacy_exit_circuit_event
'EntryClientIPCount' : { CIRCUIT_EVENT },
'EntryActiveClientIPCount' : { CIRCUIT_EVENT },
'EntryInactiveClientIPCount' : { CIRCUIT_EVENT },
'EntryClientIPActiveCircuitHistogram' : { CIRCUIT_EVENT },
'EntryClientIPInactiveCircuitHistogram' : { CIRCUIT_EVENT },

# these counters depend on stream end and circuit end
# they are updated in _handle_legacy_exit_circuit_event,
# and use data updated in _handle_stream_event

'ExitCircuitStreamHistogram' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitCircuitInterStreamCreationTime' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitWebCircuitCount' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitCircuitWebStreamHistogram' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitCircuitWebInterStreamCreationTime' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitInteractiveCircuitCount' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitCircuitInteractiveStreamHistogram' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitCircuitInteractiveInterStreamCreationTime' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitP2PCircuitCount' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitCircuitP2PStreamHistogram' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitCircuitP2PInterStreamCreationTime' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitOtherPortCircuitCount' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitCircuitOtherPortStreamHistogram' : { STREAM_EVENT, CIRCUIT_EVENT },
'ExitCircuitOtherPortInterStreamCreationTime' : { STREAM_EVENT, CIRCUIT_EVENT },

# these counters depend on connection close

# simple connection counts
'EntryConnectionCount' : { CONNECTION_EVENT },
'NonEntryConnectionCount' : { CONNECTION_EVENT },

# connection counts based on the number of relays sharing the remote address
'EntryNoRelayOnAddressConnectionCount' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCount' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCount' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCount' : { CONNECTION_EVENT },

# byte counts
'EntryConnectionByteCount' : { CONNECTION_EVENT },
'NonEntryConnectionByteCount' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionByteCount' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionByteCount' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionByteCount' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionByteCount' : { CONNECTION_EVENT },

'EntryConnectionInboundByteCount' : { CONNECTION_EVENT },
'NonEntryConnectionInboundByteCount' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionInboundByteCount' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionInboundByteCount' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionInboundByteCount' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionInboundByteCount' : { CONNECTION_EVENT },

'EntryConnectionOutboundByteCount' : { CONNECTION_EVENT },
'NonEntryConnectionOutboundByteCount' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionOutboundByteCount' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionOutboundByteCount' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionOutboundByteCount' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionOutboundByteCount' : { CONNECTION_EVENT },

# byte histograms per connection
'EntryConnectionByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionByteHistogram' : { CONNECTION_EVENT },

'EntryConnectionInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionInboundByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionInboundByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionInboundByteHistogram' : { CONNECTION_EVENT },

'EntryConnectionOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionOutboundByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionOutboundByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionOutboundByteHistogram' : { CONNECTION_EVENT },

# circuit counts
'EntryConnectionCircuitCount' : { CONNECTION_EVENT },
'NonEntryConnectionCircuitCount' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCircuitCount' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCircuitCount' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCircuitCount' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCircuitCount' : { CONNECTION_EVENT },

'EntryConnectionInboundCircuitCount' : { CONNECTION_EVENT },
'NonEntryConnectionInboundCircuitCount' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionInboundCircuitCount' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionInboundCircuitCount' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionInboundCircuitCount' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionInboundCircuitCount' : { CONNECTION_EVENT },

'EntryConnectionOutboundCircuitCount' : { CONNECTION_EVENT },
'NonEntryConnectionOutboundCircuitCount' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionOutboundCircuitCount' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionOutboundCircuitCount' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionOutboundCircuitCount' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionOutboundCircuitCount' : { CONNECTION_EVENT },

# circuit count histograms by connection
'EntryConnectionCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCircuitHistogram' : { CONNECTION_EVENT },

'EntryConnectionInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionInboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionInboundCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionInboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryConnectionOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionOutboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionOutboundCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionOutboundCircuitHistogram' : { CONNECTION_EVENT },

# connection lifetime histograms
'EntryConnectionLifeTime' : { CONNECTION_EVENT },
'NonEntryConnectionLifeTime' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionLifeTime' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionLifeTime' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionLifeTime' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionLifeTime' : { CONNECTION_EVENT },

# the number of simultaneous connections from the same IP address as a histogram
'EntryConnectionOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionOverlapHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionOverlapHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionOverlapHistogram' : { CONNECTION_EVENT },

# histograms for country codes that match the first list specified
# byte histograms per connection
'EntryConnectionCountryMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchByteHistogram' : { CONNECTION_EVENT },

'EntryConnectionCountryMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchInboundByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchInboundByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchInboundByteHistogram' : { CONNECTION_EVENT },

'EntryConnectionCountryMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchOutboundByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchOutboundByteHistogram' : { CONNECTION_EVENT },

# circuit count histograms by connection
'EntryConnectionCountryMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchCircuitHistogram' : { CONNECTION_EVENT },

'EntryConnectionCountryMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchInboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchInboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryConnectionCountryMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },

# connection lifetime histograms
'EntryConnectionCountryMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchLifeTime' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchLifeTime' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchLifeTime' : { CONNECTION_EVENT },

# the number of simultaneous connections from the same IP address as a histogram
'EntryConnectionCountryMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchOverlapHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchOverlapHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchOverlapHistogram' : { CONNECTION_EVENT },

# histograms for country codes that don't match the first list specified
# byte histograms per connection
'EntryConnectionCountryNoMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryNoMatchByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryNoMatchByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryNoMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryNoMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryNoMatchByteHistogram' : { CONNECTION_EVENT },

'EntryConnectionCountryNoMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryNoMatchInboundByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryNoMatchInboundByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryNoMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryNoMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryNoMatchInboundByteHistogram' : { CONNECTION_EVENT },

'EntryConnectionCountryNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },

# circuit count histograms by connection
'EntryConnectionCountryNoMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryNoMatchCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryNoMatchCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryNoMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryNoMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryNoMatchCircuitHistogram' : { CONNECTION_EVENT },

'EntryConnectionCountryNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryConnectionCountryNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },

# connection lifetime histograms
'EntryConnectionCountryNoMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryConnectionCountryNoMatchLifeTime' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryNoMatchLifeTime' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryNoMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryNoMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryNoMatchLifeTime' : { CONNECTION_EVENT },

# the number of simultaneous connections from the same IP address as a histogram
'EntryConnectionCountryNoMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionCountryNoMatchOverlapHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryNoMatchOverlapHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryNoMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryNoMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryNoMatchOverlapHistogram' : { CONNECTION_EVENT },

# count lists for country codes that match each list
# the final bin is used for country codes that don't match any list
# simple connection counts
'EntryConnectionCountryMatchCountList' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchCountList' : { CONNECTION_EVENT },

# connection counts based on the number of relays sharing the remote address
'EntryNoRelayOnAddressConnectionCountryMatchCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchCountList' : { CONNECTION_EVENT },

# byte counts
'EntryConnectionCountryMatchByteCountList' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchByteCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchByteCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchByteCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchByteCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchByteCountList' : { CONNECTION_EVENT },

'EntryConnectionCountryMatchInboundByteCountList' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchInboundByteCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchInboundByteCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchInboundByteCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchInboundByteCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchInboundByteCountList' : { CONNECTION_EVENT },

'EntryConnectionCountryMatchOutboundByteCountList' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchOutboundByteCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchOutboundByteCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchOutboundByteCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchOutboundByteCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchOutboundByteCountList' : { CONNECTION_EVENT },

# circuit counts
'EntryConnectionCountryMatchCircuitCountList' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchCircuitCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchCircuitCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchCircuitCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchCircuitCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchCircuitCountList' : { CONNECTION_EVENT },

'EntryConnectionCountryMatchInboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchInboundCircuitCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchInboundCircuitCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchInboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchInboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchInboundCircuitCountList' : { CONNECTION_EVENT },

'EntryConnectionCountryMatchOutboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryConnectionCountryMatchOutboundCircuitCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionCountryMatchOutboundCircuitCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionCountryMatchOutboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionCountryMatchOutboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionCountryMatchOutboundCircuitCountList' : { CONNECTION_EVENT },

# histograms for AS numbers that match the first list specified
# byte histograms per connection
'EntryConnectionASMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchByteHistogram' : { CONNECTION_EVENT },

'EntryConnectionASMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchInboundByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchInboundByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchInboundByteHistogram' : { CONNECTION_EVENT },

'EntryConnectionASMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchOutboundByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchOutboundByteHistogram' : { CONNECTION_EVENT },

# circuit count histograms by connection
'EntryConnectionASMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchCircuitHistogram' : { CONNECTION_EVENT },

'EntryConnectionASMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchInboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchInboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryConnectionASMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },

# connection lifetime histograms
'EntryConnectionASMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchLifeTime' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchLifeTime' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchLifeTime' : { CONNECTION_EVENT },

# the number of simultaneous connections from the same IP address as a histogram
'EntryConnectionASMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchOverlapHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchOverlapHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchOverlapHistogram' : { CONNECTION_EVENT },

# histograms for AS numbers that don't match the first list specified
# byte histograms per connection
'EntryConnectionASNoMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASNoMatchByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASNoMatchByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASNoMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASNoMatchByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASNoMatchByteHistogram' : { CONNECTION_EVENT },

'EntryConnectionASNoMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASNoMatchInboundByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASNoMatchInboundByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASNoMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASNoMatchInboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASNoMatchInboundByteHistogram' : { CONNECTION_EVENT },

'EntryConnectionASNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASNoMatchOutboundByteHistogram' : { CONNECTION_EVENT },

# circuit count histograms by connection
'EntryConnectionASNoMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASNoMatchCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASNoMatchCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASNoMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASNoMatchCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASNoMatchCircuitHistogram' : { CONNECTION_EVENT },

'EntryConnectionASNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASNoMatchInboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryConnectionASNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASNoMatchOutboundCircuitHistogram' : { CONNECTION_EVENT },

# connection lifetime histograms
'EntryConnectionASNoMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryConnectionASNoMatchLifeTime' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASNoMatchLifeTime' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASNoMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASNoMatchLifeTime' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASNoMatchLifeTime' : { CONNECTION_EVENT },

# the number of simultaneous connections from the same IP address as a histogram
'EntryConnectionASNoMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryConnectionASNoMatchOverlapHistogram' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASNoMatchOverlapHistogram' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASNoMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASNoMatchOverlapHistogram' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASNoMatchOverlapHistogram' : { CONNECTION_EVENT },

# count lists for AS numbers that match each list
# the final bin is used for AS numbers that don't match any list
# simple connection counts
'EntryConnectionASMatchCountList' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchCountList' : { CONNECTION_EVENT },

# connection counts based on the number of relays sharing the remote address
'EntryNoRelayOnAddressConnectionASMatchCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchCountList' : { CONNECTION_EVENT },

# byte counts
'EntryConnectionASMatchByteCountList' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchByteCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchByteCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchByteCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchByteCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchByteCountList' : { CONNECTION_EVENT },

'EntryConnectionASMatchInboundByteCountList' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchInboundByteCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchInboundByteCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchInboundByteCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchInboundByteCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchInboundByteCountList' : { CONNECTION_EVENT },

'EntryConnectionASMatchOutboundByteCountList' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchOutboundByteCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchOutboundByteCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchOutboundByteCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchOutboundByteCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchOutboundByteCountList' : { CONNECTION_EVENT },

# circuit counts
'EntryConnectionASMatchCircuitCountList' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchCircuitCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchCircuitCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchCircuitCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchCircuitCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchCircuitCountList' : { CONNECTION_EVENT },

'EntryConnectionASMatchInboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchInboundCircuitCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchInboundCircuitCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchInboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchInboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchInboundCircuitCountList' : { CONNECTION_EVENT },

'EntryConnectionASMatchOutboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryConnectionASMatchOutboundCircuitCountList' : { CONNECTION_EVENT },

'EntryNoRelayOnAddressConnectionASMatchOutboundCircuitCountList' : { CONNECTION_EVENT },
'EntryRelayOnAddressConnectionASMatchOutboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryNoRelayOnAddressConnectionASMatchOutboundCircuitCountList' : { CONNECTION_EVENT },
'NonEntryRelayOnAddressConnectionASMatchOutboundCircuitCountList' : { CONNECTION_EVENT },

# these counters depend on the HSDir store event

# HSDir Store /Add/Reject /Cached/Uncached Count/{Descriptor,Intro}Byte{Count,Histogram}/ReasonCountList

'HSDirStoreCount' : { HSDIR_STORE_EVENT },
'HSDirStoreDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreReasonCountList' : { HSDIR_STORE_EVENT },
'HSDirStoreCachedCount' : { HSDIR_STORE_EVENT },
'HSDirStoreCachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreCachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreCachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreCachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreCachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDirStoreUncachedCount' : { HSDIR_STORE_EVENT },
'HSDirStoreUncachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreUncachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreUncachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreUncachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreUncachedReasonCountList' : { HSDIR_STORE_EVENT },

'HSDirStoreAddCount' : { HSDIR_STORE_EVENT },
'HSDirStoreAddDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreAddDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreAddIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreAddIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreAddReasonCountList' : { HSDIR_STORE_EVENT },
'HSDirStoreAddCachedCount' : { HSDIR_STORE_EVENT },
'HSDirStoreAddCachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreAddCachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreAddCachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreAddCachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreAddCachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDirStoreAddUncachedCount' : { HSDIR_STORE_EVENT },
'HSDirStoreAddUncachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreAddUncachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreAddUncachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreAddUncachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreAddUncachedReasonCountList' : { HSDIR_STORE_EVENT },

'HSDirStoreRejectCount' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectReasonCountList' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectCachedCount' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectCachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectCachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectCachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectCachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectCachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectUncachedCount' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectUncachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectUncachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectUncachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectUncachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDirStoreRejectUncachedReasonCountList' : { HSDIR_STORE_EVENT },


'HSDir2StoreCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreNoClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreNoClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreNoClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreNoClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreNoClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreNoClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreNoClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreNoClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreNoClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedNoClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedNoClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedNoClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedNoClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedNoClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedNoClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedNoClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedNoClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreCachedNoClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedNoClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedNoClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedNoClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedNoClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedNoClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedNoClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedNoClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedNoClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreUncachedNoClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },

'HSDir2StoreAddCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddNoClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddNoClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddNoClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddNoClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddNoClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddNoClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddNoClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddNoClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddNoClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedNoClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedNoClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedNoClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedNoClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedNoClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedNoClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedNoClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedNoClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddCachedNoClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedNoClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedNoClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedNoClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedNoClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedNoClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedNoClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedNoClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedNoClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreAddUncachedNoClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },

'HSDir2StoreRejectCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectNoClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectNoClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectNoClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectNoClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectNoClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectNoClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectNoClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectNoClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectNoClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedNoClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedNoClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedNoClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedNoClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedNoClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedNoClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedNoClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedNoClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectCachedNoClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedNoClientAuthCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedNoClientAuthDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedNoClientAuthDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedNoClientAuthIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedNoClientAuthIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedNoClientAuthIntroPointHistogram' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedNoClientAuthUploadDelayTime' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedNoClientAuthReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir2StoreRejectUncachedNoClientAuthOnionAddressCountList' : { HSDIR_STORE_EVENT },


'HSDir3StoreCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreRevisionHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir3StoreCachedCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreCachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreCachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreCachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreCachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreCachedRevisionHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreCachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir3StoreUncachedCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreUncachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreUncachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreUncachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreUncachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreUncachedRevisionHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreUncachedReasonCountList' : { HSDIR_STORE_EVENT },

'HSDir3StoreAddCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddRevisionHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddCachedCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddCachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddCachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddCachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddCachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddCachedRevisionHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddCachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddUncachedCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddUncachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddUncachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddUncachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddUncachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddUncachedRevisionHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreAddUncachedReasonCountList' : { HSDIR_STORE_EVENT },

'HSDir3StoreRejectCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectRevisionHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectCachedCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectCachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectCachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectCachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectCachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectCachedRevisionHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectCachedReasonCountList' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectUncachedCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectUncachedDescriptorByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectUncachedDescriptorByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectUncachedIntroByteCount' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectUncachedIntroByteHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectUncachedRevisionHistogram' : { HSDIR_STORE_EVENT },
'HSDir3StoreRejectUncachedReasonCountList' : { HSDIR_STORE_EVENT },

# descriptor fetch counters

# HSDir Fetch /Cached/Uncached Count/{Descriptor,Intro}Byte{Count,Histogram}/ReasonCountList

'HSDirFetchCount' : { HSDIR_FETCH_EVENT },
'HSDirFetchDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDirFetchDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDirFetchIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDirFetchIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDirFetchReasonCountList' : { HSDIR_FETCH_EVENT },

'HSDirFetchCachedCount' : { HSDIR_FETCH_EVENT },
'HSDirFetchCachedDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDirFetchCachedDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDirFetchCachedIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDirFetchCachedIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDirFetchCachedReasonCountList' : { HSDIR_FETCH_EVENT },

'HSDirFetchUncachedCount' : { HSDIR_FETCH_EVENT },
'HSDirFetchUncachedDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDirFetchUncachedDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDirFetchUncachedIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDirFetchUncachedIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDirFetchUncachedReasonCountList' : { HSDIR_FETCH_EVENT },

# HSDir 2 Fetch /Cached/Uncached /ClientAuth/NoClientAuth Count/{Descriptor,Intro}Byte{Count,Histogram}/IntroPointHistogram/ReasonCountList/OnionAddressCountList

'HSDir2FetchCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchIntroPointHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchReasonCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchOnionAddressCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchClientAuthCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchClientAuthDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchClientAuthDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchClientAuthIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchClientAuthIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchClientAuthIntroPointHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchClientAuthReasonCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchClientAuthOnionAddressCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchNoClientAuthCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchNoClientAuthDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchNoClientAuthDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchNoClientAuthIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchNoClientAuthIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchNoClientAuthIntroPointHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchNoClientAuthReasonCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchNoClientAuthOnionAddressCountList' : { HSDIR_FETCH_EVENT },

'HSDir2FetchCachedCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedIntroPointHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedReasonCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedOnionAddressCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedClientAuthCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedClientAuthDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedClientAuthDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedClientAuthIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedClientAuthIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedClientAuthIntroPointHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedClientAuthReasonCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedClientAuthOnionAddressCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedNoClientAuthCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedNoClientAuthDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedNoClientAuthDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedNoClientAuthIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedNoClientAuthIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedNoClientAuthIntroPointHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedNoClientAuthReasonCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchCachedNoClientAuthOnionAddressCountList' : { HSDIR_FETCH_EVENT },

'HSDir2FetchUncachedCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedIntroPointHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedReasonCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedOnionAddressCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedClientAuthCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedClientAuthDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedClientAuthDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedClientAuthIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedClientAuthIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedClientAuthIntroPointHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedClientAuthReasonCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedClientAuthOnionAddressCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedNoClientAuthCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedNoClientAuthDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedNoClientAuthDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedNoClientAuthIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedNoClientAuthIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedNoClientAuthIntroPointHistogram' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedNoClientAuthReasonCountList' : { HSDIR_FETCH_EVENT },
'HSDir2FetchUncachedNoClientAuthOnionAddressCountList' : { HSDIR_FETCH_EVENT },

# HSDir 3 Fetch /Cached/Uncached Count/{Descriptor,Intro}Byte{Count,Histogram}/RevisionHistogram/ReasonCountList

'HSDir3FetchCount' : { HSDIR_FETCH_EVENT },
'HSDir3FetchDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir3FetchDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir3FetchIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir3FetchIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir3FetchRevisionHistogram' : { HSDIR_FETCH_EVENT },
'HSDir3FetchReasonCountList' : { HSDIR_FETCH_EVENT },

'HSDir3FetchCachedCount' : { HSDIR_FETCH_EVENT },
'HSDir3FetchCachedDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir3FetchCachedDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir3FetchCachedIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir3FetchCachedIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir3FetchCachedRevisionHistogram' : { HSDIR_FETCH_EVENT },
'HSDir3FetchCachedReasonCountList' : { HSDIR_FETCH_EVENT },

'HSDir3FetchUncachedCount' : { HSDIR_FETCH_EVENT },
'HSDir3FetchUncachedDescriptorByteCount' : { HSDIR_FETCH_EVENT },
'HSDir3FetchUncachedDescriptorByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir3FetchUncachedIntroByteCount' : { HSDIR_FETCH_EVENT },
'HSDir3FetchUncachedIntroByteHistogram' : { HSDIR_FETCH_EVENT },
'HSDir3FetchUncachedRevisionHistogram' : { HSDIR_FETCH_EVENT },
'HSDir3FetchUncachedReasonCountList' : { HSDIR_FETCH_EVENT },

# the sanity check counter doesn't depend on any events
DEFAULT_DUMMY_COUNTER_NAME : set(),
}

def register_dynamic_counter(counter_name, counter_events):
    '''
    Register counter_name as a counter which uses the events in counter_events.
    If counter_name is already a registered counter, updates the list of events
    for counter.
    This should be called before the counters are checked:
    - in the Tally Server, early in refresh_config,
    - in PrivCountClient, early in check_start_config
      (PrivCountClient is a parent class of Data Collector and Share Keeper)
    Any event updates are applied the next time the data collector starts a
    collection phase.
    Logs a message and ignores unknown events.
    '''
    event_set = set()
    for event in counter_events:
        if event in get_valid_events():
            event_set.add(event)
        else:
            logging.warning("Ignoring unknown event {} for dynamic counter {}"
                            .format(event, counter_name))
    PRIVCOUNT_COUNTER_EVENTS[counter_name] = event_set

def get_valid_counters():
    '''
    Return a set containing the name of each privcount counter, in titlecase.
    (Or whatever the canonical case of the counter name is.)
    '''
    counter_set = set(PRIVCOUNT_COUNTER_EVENTS.keys())
    # we can't check case consistency, so just return the set
    return counter_set

def get_events_for_counter(counter):
    '''
    Return the set of events required by counter
    '''
    # when you add an event, but forget to update the table above,
    # you will get an error here
    logging.debug("Finding events for counter: '{}'".format(counter))
    try:
        event_set = PRIVCOUNT_COUNTER_EVENTS[counter]
    except KeyError as e:
        logging.error("Missing events for counter: '{}'".format(counter))
        raise
    assert check_event_set_valid(event_set)
    return event_set

def get_events_for_counters(counter_list):
    '''
    Return the set of events required by at least one of the counters in
    counter_list.
    '''
    event_set = set()
    if counter_list is not None:
        for counter in counter_list:
            counter_events = get_events_for_counter(counter)
            event_set = event_set.union(counter_events)
    assert check_event_set_valid(event_set)
    return event_set

def get_events_for_known_counters():
    '''
    Return the set of events required by at least one of the counters we know
    about.
    '''
    return get_events_for_counters(PRIVCOUNT_COUNTER_EVENTS.keys())

def get_circuit_sample_events():
    '''
    Return the set of events affected by circuit_sample_rate.
    '''
    event_set = { CELL_EVENT,
                  BYTES_EVENT,
                  STREAM_EVENT,
                  CIRCUIT_EVENT,
                  # Not affected
                  #CONNECTION_EVENT,
                  #HSDIR_STORE_EVENT,
                  # Unused events
                  DNS_EVENT,
                  LEGACY_CIRCUIT_EVENT,
                  }
    return event_set

def is_circuit_sample_counter(counter):
    '''
    If counter uses an event affected by circuit_sample_rate, return True.
    Otherwise, return False.
    '''
    counter_events = get_events_for_counter(counter)
    circuit_sample_events = get_circuit_sample_events()
    common_events = counter_events.intersection(circuit_sample_events)
    return len(common_events) > 0

def are_events_expected(counter_list, relay_flag_list):
    '''
    Return True if we expect to receive regular events while collecting
    counter_list, on a relay with the consensus flags in relay_flag_list.
    relay_flag_list must be a list, not a string.
    Return False if we don't expect to receive events regularly.
    '''
    # It really does need to be a list
    if isinstance(relay_flag_list, (str, unicode)):
        relay_flag_list = relay_flag_list.split()
    # no counters
    if counter_list is None or len(counter_list) == 0:
        return False
    event_list = get_events_for_counters(counter_list)
    # no events: ZeroCount only
    if event_list is None or len(event_list) == 0:
        return False
    has_entry = "Guard" in relay_flag_list
    has_exit = "Exit" in relay_flag_list
    # relay_flag_list must be a list to avoid a substring match
    has_hsdir2 = "HSDir" in relay_flag_list
    has_hsdir3 = "HSDir3" in relay_flag_list
    for counter_name in counter_list:
        if has_entry and counter_name.startswith("Entry"):
            return True
        if has_exit and counter_name.startswith("Exit"):
            return True
        if has_hsdir2 and counter_name.startswith("HSDir2"):
            return True
        if has_hsdir3 and counter_name.startswith("HSDir3"):
            return True
    # no matching counters and flags
    return False

def check_counter_names(counters):
    '''
    Check that each counter's name is in the set of valid counter names.
    Returns False if any counter name is unknown, True if all are known.
    '''
    # sort names alphabetically, so the logs are in a sensible order
    for counter_name in sorted(counters.keys()):
        if counter_name not in get_valid_counters():
            logging.warning("counter name {} is unknown"
                            .format(counter_name))
            return False
    return True

def count_bins(counters):
    '''
    Returns the total number of bins in counters.
    '''
    return sum([len(counter_config['bins'])
                for counter_config in counters.values()])

def check_bin_count_matches_name(bins):
    '''
    Check that counter names that end in "Count" have a single bin, and
    counter names that end in anything else have multiple bins.
    '''
    # sort names alphabetically, so the logs are in a sensible order
    for key in sorted(bins.keys()):
        bin_count = len(bins[key]['bins'])
        # handle template counters by stripping the non-template part
        key_template, _, _ = key.partition("_")
        # the TrafficModel LogDelayTime counters are single bin
        if key_template.endswith("Count") or key_template.endswith("LogDelayTime"):
            if bin_count != 1:
                logging.warning("counter {} ends in Count, but has {} bins: {}"
                                .format(key, bin_count, bins[key]))
                return False
        else: # Histogram, Ratio, LifeTime, DelayTime, CountList, ...
            if bin_count <= 1:
                logging.warning("counter {} does not end in Count, but has {} bins: {}"
                                .format(key, bin_count, bins[key]))
                return False
    return True

def check_bins_config(bins, allow_unknown_counters=False):
    '''
    Check that bins are non-overlapping.
    Returns True if all bins are non-overlapping, and False if any overlap.
    If allow_unknown_counters is False, also check that all counter names are
    in the set of known counter names for this PrivCount version, returning
    False if there are any unknown counters.
    Raises an exception if any counter does not have bins, or if any bin does
    not have a lower and upper bound
    '''
    if not allow_unknown_counters:
        if not check_counter_names(bins):
            return False
        # unknown counters may have different rules for bin counts
        if not check_bin_count_matches_name(bins):
            return False
    # sort names alphabetically, so the logs are in a sensible order
    for key in sorted(bins.keys()):
        # this sorts the bins by the first element in ascending order
        # (if the first elements are equal, the bins are sorted by the second
        # element)
        sorted_bins = sorted(bins[key]['bins'])
        prev_bin = None
        for bin in sorted_bins:
            # bins are an array [l, u, c], where c counts values such that:
            # l <= value < u
            # c is optional, and is ignored by this code
            l = bin[0]
            u = bin[1]
            # check for inverted bounds
            if l >= u:
                logging.warning("bin {} in counter {} will never count any values, because its lower bound is greater than or equal to its upper bound"
                                .format(bin, key))
                return False
            # make sure we have a bin to compare to
            if prev_bin is not None:
                prev_l = prev_bin[0]
                prev_u = prev_bin[1]
                # two sorted bins overlap if:
                # - their lower bounds are equal, or
                # - the upper bound of a bin is greater than the lower bound
                #   of the next bin
                if prev_l == l:
                    logging.warning("bin {} in counter {} overlaps bin {}: their lower bounds are equal"
                                    .format(prev_bin, key, bin))
                    return False
                elif prev_u > l:
                    logging.warning("bin {} in counter {} overlaps bin {}: the first bin's upper bound is greater than the second bin's lower bound"
                                    .format(prev_bin, key, bin))
                    return False
            prev_bin = bin
    return True

def check_sigmas_config(sigmas, allow_unknown_counters=False):
    '''
    Check that each sigma value in sigmas is valid.
    Returns True if all sigma values are valid, and False if any are invalid.
    If allow_unknown_counters is False, also check that all counter names are
    in the set of known counter names for this PrivCount version, returning
    False if there are any unknown counters.
    Raises an exception if any sigma value is missing.
    '''
    if not allow_unknown_counters:
        if not check_counter_names(sigmas):
            return False
    # sort names alphabetically, so the logs are in a sensible order
    for key in sorted(sigmas.keys()):
        if sigmas[key]['sigma'] < 0.0:
            logging.warning("invalid sigma for counter {}: less than zero".format(key))
            return False
    return True

def extra_counters(first, second, first_name, second_name, action_name):
    '''
    Return the extra counter keys in first that are not in second.
    Warn about taking action_name on any missing counters.
    '''
    extra_keys = _extra_keys(first, second)
    # Log missing keys
    if len(extra_keys) > 0:
        logging.info("{} counters '{}' because they have a {}, but no {}"
                     .format(action_name, summarise_list(extra_keys),
                             first_name, second_name))

    return extra_keys

def common_counters(first, second, first_name, second_name, action_name):
    '''
    Return the counter keys shared by first and second.
    Warn about taking action_name on any missing counters.
    '''
    # ignore the extra counters return values, we just want the logging
    extra_counters(first, second, first_name, second_name, action_name)
    extra_counters(second, first, second_name, first_name, action_name)

    # return common keys
    return _common_keys(first, second)

def _skip_missing(counters, expected_subkey, detailed_source=None):
    '''
    Check that each key in counters has a subkey with the name expected_subkey.
    If any key does not have a subkey named expected_subkey, skip it and log a
    warning.
    If detailed_source is not None, use it to describe the counters.
    Otherwise, use expected_subkey.
    Returns a copy of counters with invalid keys skipped.
    '''
    if detailed_source is None:
        detailed_source = expected_subkey
    valid_counters = {}
    invalid_counters = []
    for key in sorted(counters.keys()):
        if expected_subkey in counters[key]:
            valid_counters[key] = counters[key]
        else:
            invalid_counters.append(key)
    if len(invalid_counters) > 0:
        logging.warning("ignoring counters {} because they are configured as {} counters, but they do not have any {} value"
                        .format(summarise_list(invalid_counters),
                                detailed_source, expected_subkey))
    return valid_counters

def skip_missing_bins(bins, detailed_source=None):
    '''
    Check each key in bins has a bins list.
    If any key does not have a bins list, skip it and log a warning.
    Returns a copy of counters with invalid keys skipped.
    '''
    return _skip_missing(bins, 'bins', detailed_source)

def skip_missing_sigmas(sigmas, detailed_source=None):
    '''
    Check each key in sigmas has a sigma value.
    If any key does not have a sigma, skip it and log a warning.
    Returns a copy of counters with invalid keys skipped.
    '''
    return _skip_missing(sigmas, 'sigma')

def combine_counters(bins, sigmas):
    '''
    Combine the counters in bins and sigmas, excluding any counters that are
    missing from either bins or sigmas.
    Combine the keys and values from both bins and sigmas in the output
    counters, according to what the tally server is permitted to update.
    (Both bins and sigmas are configured at the tally server.)
    Return a dictionary containing the combined keys.
    '''
    # Remove invalid counters
    bins = skip_missing_bins(bins)
    sigmas = skip_missing_sigmas(sigmas)

    # we allow the tally server to update the set of counters
    # (we can't count keys for which we don't have both bins and sigmas)
    common_keys = common_counters(bins, sigmas, 'bins', 'sigma',
                                  'ignoring')

    counters_combined = {}
    for key in common_keys:
        # skip_missing_* ensures these exist
        assert 'bins' in bins[key]
        assert 'sigma' in sigmas[key]
        # Use the values from the sigmas
        counters_combined[key] = deepcopy(sigmas[key])
        # Except for the bin values, which come from bins
        # we allow the tally server to update the bin widths
        counters_combined[key]['bins'] = deepcopy(bins[key]['bins'])
    return counters_combined

def check_combined_counters(bins, sigmas):
    '''
    Sanity check bins against sigmas.
    Returns False if:
      - the set of counters in bins and sigmas is not the same, or
      - any counter is missing bins, or
      - any counter is missing a sigma, or
      - any counter is duplicated.
    '''
    combined_counters = combine_counters(bins, sigmas)
    return (len(combined_counters) == len(bins) and
            len(combined_counters) == len(sigmas))

def check_counters_config(bins, sigmas, allow_unknown_counters=False):
    '''
    Sanity check bins and sigmas individually.
    Check that bins and sigmas have the same set of counters.
    If allow_unknown_counters is False, also check that all counter names are
    in the set of known counter names for this PrivCount version.
    '''
    return (check_bins_config(bins,
                          allow_unknown_counters=allow_unknown_counters) and
            check_sigmas_config(sigmas,
                          allow_unknown_counters=allow_unknown_counters) and
            check_combined_counters(bins, sigmas))

def float_representation_accuracy():
    '''
    When converting an exact number to a python float, the maximum possible
    proportional change in the value of the float.
    For the exact number n, converting n to a float could change the value by
    at most +/- n * float_representation_accuracy().
    Returns a floating point number representing the maximum relative increase
    or decrease in the value of the original exact number.
    '''
    # When converting an exact value to a python float, the maximum possible
    # proportional change is half the distance between one float value and the
    # next largest or smallest float value.
    # Conventiently, the distance between adjacent floats is at most the float
    # epsilon multiplied by the value of the float, as the distance between
    # adjacent floats scales as they get larger or smaller.
    # On most platforms, the float epsilon is 2 ** -53.
    return sys.float_info.epsilon/2.0

def float_string_accuracy():
    '''
    When converting a python float to a string and back, the maximum possible
    proportional change in the value of the float.
    For the float f, converting f to a string and back could change the value
    by at most +/- f * float_string_accuracy().
    Returns a floating point number representing the maximum relative increase
    or decrease in the value of the original float.
    '''
    # sys.float_info.dig is the number of significant figures that are
    # guaranteed to be preserved when converting a float to a string and
    # then back to a float (PrivCount does this when sending sigma between
    # the TS and the SKs/DCs).
    # This is based on python's float repr() rule, introduced in versions 2.7
    # and 3.1:
    # Python "displays a value based on the shortest decimal fraction that
    # rounds correctly back to the true binary value"
    # On most 32 and 64-bit platforms, sys.float_info.dig is 15 digits.
    # Therefore, the maximum change in value that can occur is the 15th digit
    # (of least significance) changing by +/- 1.
    # But we can't just multiply the original value by 10 ** -15, because
    # the (significand of the) float can have any value in [0.1, 0.999...].
    # Therefore, we need to multiply the tolerance by another 10x.
    # This gives us a tolerance of 10 ** -14 on most systems.
    return 10.0 ** (-sys.float_info.dig + 1)

def float_accuracy():
    '''
    The maximum proportional change in an exact value when converted to a
    float, then a string, then back to a float.
    For the exact number n, converting n to a float then string then float
    could change the value by at most +/- n * float_accuracy().
    Returns a floating point number representing the maximum relative increase
    or decrease in the value of the original exact number.
    '''
    # If the inaccuracies are both in the same direction, the total inaccuracy
    # is the sum of all inaccuracies
    return float_representation_accuracy() + float_string_accuracy()

class CollectionDelay(object):
    '''
    Ensures a configurable delay between rounds with different noise
    allocations.
    Usage:
    (the SKs must enforce these checks for the protocol to be secure
     the TS does these checks for convenience, the DCs for defence in depth)
    TS: configures round
        uses get_next_round_start_time() for status updates
        checks round_start_permitted() before starting collection
    DC: checks round_start_permitted() before sending blinding shares
    SK: checks round_start_permitted() before accepting blinding shares
    (round runs)
    DC: set_delay_for_stop() when round stops and counters are sent
    SK: set_delay_for_stop() when round stops and blinding shares are sent
    TS: set_delay_for_stop() when round ends successfully
    (repeat for next round, if TS has continue set in its config)
    '''

    def __init__(self):
        '''
        Initialise the noise allocations and times required to track collection
        delays.
        '''
        # The earliest noise allocation in a series of equivalent noise
        # allocations
        self.starting_noise_allocation = None
        # The end time of the successful round to use an equivalent allocation
        self.last_round_end_time = None

    DEFAULT_SIGMA_DECREASE_TOLERANCE = DEFAULT_SIGMA_TOLERANCE

    @staticmethod
    def sigma_change_needs_delay(
            previous_sigma, proposed_sigma,
            tolerance=DEFAULT_SIGMA_DECREASE_TOLERANCE,
            logging_label=None):
        '''
        Check if there should be a delay between rounds using the previous
        and proposed sigma values for the same counter.
        A counter can use two sigma values without a delay between them if:
        - The values are equal (within a small tolerance), or
        - The proposed value is greater than the previous value.
        Returns True if the sigma values need a delay, False if they do not.
        '''
        assert previous_sigma >= 0
        assert proposed_sigma >= 0
        assert tolerance >= 0
        if proposed_sigma >= previous_sigma:
            # the sigma has increased: no delay required
            return False
        elif previous_sigma - proposed_sigma <= tolerance:
            # the sigma has decreased, but not by enough to matter
            return False
        # the sigma has decreased too much - enforce a delay
        if logging_label is not None:
            logging.warning("Delaying round: proposed sigma %.2g is less than previous sigma %.2g, and not within tolerance %.2g, in counter %s",
                            proposed_sigma,
                            previous_sigma,
                            tolerance,
                            logging_label)
        return True

    @staticmethod
    def noise_change_needs_delay(
            previous_allocation, proposed_allocation,
            tolerance=DEFAULT_SIGMA_DECREASE_TOLERANCE):
        '''
        Check if there should be a delay between rounds using the previous
        and proposed noise allocations.
        Two allocations can be used without a delay between them if:
        - They have the same keys, and
        - The sigma values for those keys do not need a delay, using the
          acceptable sigma decrease tolerance.
        Returns True if the allocations need a delay, False if they do not.
        '''
        # There must be an allocation for a valid round
        assert proposed_allocation is not None
        assert tolerance >= 0
        # No delay for the first round
        if previous_allocation is None:
            return False

        # Ignore and log missing sigmas
        previous_sigmas = skip_missing_sigmas(previous_allocation['counters'],
                                              'previous sigma')
        proposed_sigmas = skip_missing_sigmas(proposed_allocation['counters'],
                                              'proposed sigma')

        # Check that we have the same set of counters
        common_sigmas = common_counters(previous_sigmas, proposed_sigmas,
                                        'previous sigma', 'proposed sigma',
                                        "can't compare sigmas on")
        if len(common_sigmas) != len(previous_sigmas):
            return True
        if len(common_sigmas) != len(proposed_sigmas):
            return True

        # check the sigma values are the same
        for key in sorted(common_sigmas):
            if CollectionDelay.sigma_change_needs_delay(
                previous_sigmas[key]['sigma'],
                proposed_sigmas[key]['sigma'],
                tolerance=tolerance,
                logging_label=key):
                return True
        return False

    def get_next_round_start_time(
            self, noise_allocation, delay_period,
            max_client_rtt=0.0,
            always_delay=False,
            tolerance=DEFAULT_SIGMA_DECREASE_TOLERANCE):
        '''
        Return the earliest time at which a round with noise allocation could
        start, where delay_period is the configurable delay.
        If always_delay is True, always delay the round by delay_period.
        (This is intended for use while testing.)
        max_client_rtt is the maximum client RTT of all clients (only used by
        the Tally Server).
        tolerance is the acceptable sigma decrease.
        '''
        # there must be a configured delay_period (or a default must be used)
        assert delay_period >= 0
        # that is, it must be boolean-coercible
        assert always_delay or not always_delay
        # there must be a noise allocation for the next round
        assert noise_allocation is not None
        assert tolerance >= 0

        noise_change_delay = self.noise_change_needs_delay(
                                      self.starting_noise_allocation,
                                      noise_allocation,
                                      tolerance=tolerance)
        needs_delay = always_delay or noise_change_delay

        if noise_change_delay:
            # if there was a change, there must have been a previous allocation
            assert self.starting_noise_allocation

        if self.last_round_end_time is None:
            # a delay is meaningless, there have been no previous successful
            # rounds
            # we can start any time
            return 0
        elif needs_delay:
            # if there was a previous round, and we need to delay after it,
            # there must have been an end time for that round
            next_start_time = self.last_round_end_time + delay_period + max_client_rtt
            return next_start_time
        else:
            # we can start any time after the last round ended
            return self.last_round_end_time

    def round_start_permitted(
            self, noise_allocation, start_time, delay_period,
            max_client_rtt=0.0,
            always_delay=False,
            tolerance=DEFAULT_SIGMA_DECREASE_TOLERANCE,
            logging_function=logging.debug):
        '''
        Check if we are permitted to start a round with noise allocation
        at start time, with the configured delay_period and max_client_rtt.
        If always_delay is True, always delay the round by delay_period.
        (This is intended for use while testing.)
        max_client_rtt is the maximum client RTT of all clients (only used by
        the Tally Server).
        tolerance is the acceptable sigma decrease.
        Return True if starting the round is permitted.
        If it is not, return False, and log a message using logging_function.
        '''
        # there must be a start time
        assert start_time >= 0
        # all the other assertions are in this function
        next_start_time = self.get_next_round_start_time(noise_allocation,
                                                         delay_period,
                                                         max_client_rtt=max_client_rtt,
                                                         always_delay=always_delay,
                                                         tolerance=tolerance)
        if start_time >= next_start_time:
            return True
        else:
            if always_delay:
                delay_reason = "we are configured to always delay"
            else:
                delay_reason = "noise allocation changed"
            logging_function("Delaying round for %s because %s",
                             format_delay_time_until(next_start_time,
                                                     'until'),
                             delay_reason)
            return False

    def set_delay_for_stop(
            self, round_successful, noise_allocation, start_time, end_time,
            delay_period,
            max_client_rtt=0.0,
            always_delay=False,
            tolerance=DEFAULT_SIGMA_DECREASE_TOLERANCE):
        '''
        Called when a round ends.
        If the new noise allocation is not equivalent to the stored noise,
        update the stored noise. Update the stored last round end time.
        No updates are performed for failed rounds.
        Log a warning if it appears that the round was started too early.
        (This can also occur if the config is changed mid-round.)
        If always_delay is True, assume the round was delayed, regardless of
        the noise allocation. (This is intended for use while testing.)
        max_client_rtt is the maximum client RTT of all clients (only used by
        the Tally Server).
        tolerance is the acceptable sigma decrease.
        '''
        # make sure we haven't violated our own preconditions
        # that is, it must be boolean-coercible
        assert round_successful or not round_successful
        assert noise_allocation is not None
        assert start_time >= 0
        assert end_time >= 0
        assert start_time < end_time
        assert delay_period >= 0
        assert always_delay or not always_delay
        assert tolerance >= 0
        # did we forget to check if we needed to delay this round?
        # warn, because this can happen if the delay is reconfigured,
        # or if another node fails a round because it starts sooner than its
        # configured delay, or if the Tally server asks for results twice
        if not self.round_start_permitted(noise_allocation,
                                          start_time,
                                          delay_period,
                                          max_client_rtt=max_client_rtt,
                                          always_delay=always_delay,
                                          tolerance=tolerance):
            expected_start = self.get_next_round_start_time(noise_allocation,
                                                            delay_period,
                                                            max_client_rtt=max_client_rtt,
                                                            always_delay=always_delay,
                                                            tolerance=tolerance)
            status = "successfully stopped" if round_successful else "stopped unexpectedly (failure or duplicate event)"
            logging.warning("Round that just {} was started {} before enforced delay elapsed. Round started {}, expected start {}."
                            .format(status,
                                    format_period(expected_start - start_time),
                                    format_elapsed_time_since(start_time,
                                                              'at'),
                                    format_elapsed_time_since(expected_start,
                                                              'at')))
        if round_successful:
            # The end time is always updated
            self.last_round_end_time = end_time
            if self.starting_noise_allocation is None or always_delay:
                # It's the first noise allocation this run, or it's a
                # noise allocation for which we've delayed collection
                self.starting_noise_allocation = noise_allocation
            elif not self.noise_change_needs_delay(
                              self.starting_noise_allocation,
                              noise_allocation,
                              tolerance=tolerance):
                # The latest noise allocation could have been used immediately
                # after the starting noise allocation.
                # Keep the starting noise allocation, so that a TS can't
                # gradually decrease the noise each round
                pass
            else:
                # It's a noise allocation from a successful round, and it's
                # different enough from the starting allocation. Assume we
                # waited for the enforced delay before the round started.
                self.starting_noise_allocation = noise_allocation

def noise(sigma, sum_of_sq, p_exit):
    '''
    Sample noise from a gussian distribution
    the distribution is over +/- sigma, scaled by the noise weight, which is
    calculated from the exit probability p_exit, and the overall sum_of_sq
    bandwidth
    returns a floating-point value between +sigma and -sigma, scaled by
    noise_weight
    '''
    sigma_i = p_exit * sigma / sqrt(sum_of_sq)
    # the noise needs to be cryptographically secure, because knowing the RNG
    # state could allow an adversary to remove the noise
    random_sample = SystemRandom().gauss(0, sigma_i)
    return random_sample

def sample(modulus):
    '''
    Sample a uniformly distributed value from the SystemRandom CSPRNG
    (uses rejection sampling to avoid bias)
    returns a long uniformly distributed in [0, modulus)
    '''
    # sanitise input
    modulus = long(modulus)
    assert modulus > 0
    # to get values up to modulus-1, we need this many bits
    sample_bit_count = (modulus-1).bit_length()
    # handle the case where modulus is 1
    if sample_bit_count == 0:
        sample_bit_count = 1
    # check the bit count is sane
    assert modulus <= 2L**sample_bit_count
    assert modulus >= 2L**(sample_bit_count-1)
    ## Unbiased sampling through rejection sampling
    while True:
        # sample that many bits
        v = SystemRandom().getrandbits(sample_bit_count)
        assert v >= 0
        assert v < 2L**sample_bit_count
        # the maximum rejection rate is 1 in 2, when modulus is 2**N + 1
        if 0L <= v < modulus:
            break
    return v

def sample_randint(a, b):
    """
    Like random.randint(), returns a random long N such that a <= N <= b.
    """
    return a + sample(b - a + 1)

def derive_blinding_factor(secret, modulus, positive=True):
    '''
    Calculate a blinding factor less than modulus, based on secret
    If secret is None, sample a blinding factor and return it
    When positive is True, returns the blinding factor, and when positive is
    False, returns the unblinding factor (the inverse value mod modulus)
    Typically called as:
      blinding   = derive_blinding_factor(None,     counter_modulus(), True)
      unblinding = derive_blinding_factor(blinding, counter_modulus(), False)
    '''
    # sanitise input
    modulus = long(modulus)
    if secret is None:
        v = sample(modulus)
    else:
        # sanitise input
        v = long(secret)
    assert v < modulus
    s0 = v if positive else modulus - v
    return s0

def adjust_count_signed(count, modulus):
    '''
    Adjust the unsigned 0 <= count < modulus, returning a signed integer
    For odd  modulus, returns { -modulus//2, ... , 0, ... , modulus//2 }
    For even modulus, returns { -modulus//2, ... , 0, ... , modulus//2 - 1 }
    The smallest positive values >= modulus//2 [- 1] become the largest
    negative values
    This is the inverse operation of x % modulus, when x is in the appropriate
    range (x % modulus always returns a positive integer when modulus is
    positive)
    '''
    # sanitise input
    count = long(count)
    modulus = long(modulus)
    # sanity check input
    assert count < modulus
    # When implementing this adjustment,
    # { 0, ... , (modulus + 1)//2 - 1}  is interpreted as that value,
    # { (modulus + 1)//2, ... , modulus - 1 } is interpreted as
    # that value minus modulus, or
    # { (modulus + 1)//2 - modulus, ... , modulus - 1 - modulus }
    #
    # For odd modulus, (modulus + 1)//2 rounds up to modulus//2 + 1, so the
    # positive case simplifies to:
    # { 0, ... , modulus//2 + 1 - 1 }
    # { 0, ... , modulus//2 }
    # and because modulus == modulus//2 + modulus//2 + 1 for odd modulus, the
    # negative case simplifies to:
    # { modulus//2 + 1 - modulus//2 - modulus//2 - 1, ... ,
    #   modulus - 1 - modulus}
    # { -modulus//2, ... , -1 }
    # Odd modulus has the same number of values above and below 0:
    # { -modulus//2, ... , 0, ... , modulus//2 }
    #
    # For even modulus, (modulus+1)//2 rounds down to modulus//2, so the
    # positive case simplifies to:
    # { 0, ... , modulus//2 - 1 }
    # and because modulus == modulus//2 + modulus//2 for even modulus, the
    # negative case simplifies to:
    # { modulus//2 - modulus//2 - modulus//2, ... , modulus - 1 - modulus}
    # { -modulus//2, ... , -1 }
    # Even modulus has the 1 more value below 0 than above it:
    # { -modulus//2, ... , 0, ... , modulus//2 - 1 }
    # This is equivalent to signed two's complement, if modulus is an integral
    # power of two
    if count >= ((modulus + 1L) // 2L):
        signed_count = count - modulus
    else:
        signed_count = count
    # sanity check output
    assert signed_count >= -modulus//2L
    if modulus % 2L == 1L:
        # odd case
        assert signed_count <= modulus//2L
    else:
        # even case
        assert signed_count <= modulus//2L - 1L
    return signed_count

class SecureCounters(object):
    '''
    securely count any number of labels
    counters should be in the form like this:
    {
      'CircuitCellsInOutRatio': {
        'bins':
        [
          [0.0, 0.1],
          [0.1, 0.25],
          [0.25, 0.5],
          [0.5, 0.75],
          [0.75, 0.9],
          [0.9, 1.0],
          [1.0, float('inf')],
        ],
        'sigma': 2090007.68996
      },
      'EntryCircuitInboundCellHistogram': {
        'bins':
        [
          [0.0, 512.0],
          [512.0, 1024.0],
          [1024.0, 2048.0],
          [2048.0, 4096.0],
          [4096.0, float('inf')],
        ],
        'sigma': 2090007.68996
      }
    }
    All of data collectors, share keepers, and tally server use this to store
    counters.
    It is used approximately like this:

    data collector:
    init(), generate_blinding_shares(), detach_blinding_shares(),
    generate_noise(), increment()[repeated],
    detach_counts()
    the blinding shares are sent to each share keeper
    the counts are sent to the tally server at the end

    share keeper:
    init(), import_blinding_share()[repeated], detach_counts()
    import..() uses the shares from each data collector
    the counts are sent to the tally server at the end

    tally server:
    init(), tally_counters(), detach_counts()
    tally..() uses the counts received from all of the data collectors and
    share keepers
    this produces the final, unblinded, noisy counts of the privcount process

    see privcount/test/test_counters.py for some test cases
    '''

    def __init__(self, counters, modulus, require_generate_noise=True):
        '''
        deepcopy counters and initialise each counter to 0L
        cast modulus to long and store it
        If require_generate_noise is True, assert if we did not add noise
        before detaching the counters
        '''
        self.counters = deepcopy(counters)
        self.modulus = long(modulus)
        self.shares = None
        self.is_noise_pending = require_generate_noise

        # initialize all counters to 0L
        # counters use unlimited length integers to avoid overflow
        for key in self.counters:
            assert('bins' in self.counters[key])
            for item in self.counters[key]['bins']:
                assert len(item) == 2
                # bin is now, e.g.: [0.0, 512.0, 0L] for bin_left, bin_right,
                # count
                item.append(0L)

        # take a copy of the zeroed counters to use when generating blinding
        # factors
        self.zero_counters = deepcopy(self.counters)

    def _check_counter(self, counter):
        '''
        Check that the keys and bins in counter match self.counters
        Also check that each bin has a count.
        If these checks pass, return True. Otherwise, return False.
        '''
        for key in self.counters:
            if key not in counter:
                return False
            # disregard sigma, it's only required at the data collectors
            if 'bins' not in counter[key]:
                return False
            num_bins = len(self.counters[key]['bins'])
            if num_bins == 0:
                return False
            if num_bins != len(counter[key]['bins']):
                return False
            for i in xrange(num_bins):
                tally_item = counter[key]['bins'][i]
                if len(tally_item) != 3:
                    return False
        return True

    def _derive_all_counters(self, blinding_factors, positive):
        '''
        If blinding_factors is None, generate and apply a counters structure
        containing uniformly random blinding factors.
        Otherwise, apply the passed blinding factors.
        If positive is True, apply blinding factors. Otherwise, apply
        unblinding factors.
        Returns the applied (un)blinding factors, or None on error.
        '''
        # if there are no blinding_factors, initialise them to zero
        generate_factors = False
        if blinding_factors is None:
            blinding_factors = deepcopy(self.zero_counters)
            generate_factors = True

        # validate that the counter data structures match
        if not self._check_counter(blinding_factors):
            return None

        # determine the blinding factors
        for key in blinding_factors:
            for item in blinding_factors[key]['bins']:
                if generate_factors:
                    original_factor = None
                else:
                    original_factor = long(item[2])
                blinding_factor = derive_blinding_factor(original_factor,
                                                         self.modulus,
                                                         positive=positive)
                item[2] = blinding_factor

        # add the blinding factors to the counters
        self._tally_counter(blinding_factors)

        # return the applied blinding factors
        return blinding_factors

    def _blind(self):
        '''
        Generate and apply a counters structure containing uniformly random
        blinding factors.
        Returns the generated blinding factors.
        '''
        generated_counters = self._derive_all_counters(None, True)
        # since we generate blinding factors based on our own inputs, a
        # failure here is a programming bug
        assert generated_counters is not None
        return generated_counters

    def _unblind(self, blinding_factors):
        '''
        Generate unblinding factors from blinding_factors, and apply them to
        self.counters.
        Returns the applied unblinding factors.
        '''
        # since we generate unblinding factors based on network input, a
        # failure here should be logged, and the counters ignored
        return self._derive_all_counters(blinding_factors, False)

    def generate_blinding_shares(self, uids):
        '''
        Generate and apply blinding factors for each counter and share keeper
        uid.
        '''
        self.shares = {}
        for uid in uids:
            # add blinding factors to all of the counters
            blinding_factors = self._blind()
            # the caller can add additional annotations to this dictionary
            self.shares[uid] = {'secret': blinding_factors, 'sk_uid': uid}

    def generate_noise(self, noise_weight):
        '''
        Generate and apply noise for each counter.
        '''
        # generate noise for each counter independently
        noise_values = deepcopy(self.zero_counters)
        for key in noise_values:
            for item in noise_values[key]['bins']:
                sigma = noise_values[key]['sigma']
                sampled_noise = noise(sigma, 1, noise_weight)
                # exact halfway values are rounded towards even integers
                # values over 2**53 are not integer-accurate
                # but we don't care, because it's just noise
                item[2] = long(round(sampled_noise))

        # add the noise to each counter
        self._tally_counter(noise_values)
        self.is_noise_pending = False

    def detach_blinding_shares(self):
        '''
        Deletes this class' reference to self.shares.
        Does not securely delete, as python does not have secure delete.
        Detaches and returns the value of self.shares.
        Typically, the caller then uses encrypt() on the returned shares.
        '''
        shares = self.shares
        # TODO: secure delete
        # del only deletes the reference binding
        # deallocation is implementation-dependent
        del self.shares
        self.shares = None
        return shares

    def import_blinding_share(self, share):
        '''
        Generate and apply reverse blinding factors to all of the counters.
        If encrypted, these blinding factors must be decrypted and decoded by
        the caller using decrypt(), before calling this function.
        Returns True if unblinding was successful, and False otherwise.
        '''
        unblinding_factors = self._unblind(share['secret'])
        if unblinding_factors is None:
            return False
        return True


    SINGLE_BIN = float('nan')
    '''
    A placeholder for the bin value of a counter with a single bin.
    This constant must be outside the range of every possible counter.
    '''

    @staticmethod
    def is_single_bin_value(value):
        if isnan(SecureCounters.SINGLE_BIN):
            return isnan(value)
        else:
            return SecureCounters.SINGLE_BIN == value

    @staticmethod
    def is_in_bin(bin_min, bin_max, bin_value):
        '''
        Is bin_value between bin_min and bin_max?
        bin_min is always inclusive. bin_max is exclusive, except when it is
        inf, it includes inf.
        '''
        # make everything float for consistent comparisons
        bin_min = float(bin_min)
        bin_max = float(bin_max)
        bin_value = float(bin_value)
        if bin_value >= bin_min:
            # any value is <= inf, so we don't need to check if bin_value is inf
            if bin_value < bin_max or bin_max == float('inf'):
                return True
        return False

    def increment(self, counter_name, bin=SINGLE_BIN, inc=1):
        '''
        Increment a bin in counter counter_name by inc.
        Uses is_in_bin() to work out which bin to increment.
        Example:
            secure_counters.increment('ExampleHistogram',
                                      bin=25,
                                      inc=1)

        If there is only one bin for the counter, you must pass SINGLE_BIN
        for bin:
            secure_counters.increment('ExampleCount',
                                      bin=SINGLE_BIN,
                                      inc=1)
        '''
        if self.counters is not None and counter_name in self.counters:
            # check that we have the right types, and that we're not losing
            # precision
            bin = float(bin)
            if float(inc) != float(int(inc)):
                logging.warning("Ignoring fractional part of counter {} bin {} increment {}: {}"
                                .format(counter_name, bin, inc,
                                        float(inc) - float(int(inc))))
                assert float(inc) == float(int(inc))
            inc = int(inc)
            # You must pass SINGLE_BIN if counter_name is a single bin
            if len(self.counters[counter_name]['bins']) == 1:
                assert(SecureCounters.is_single_bin_value(bin))
                bin = 1.0
            else:
                assert(not SecureCounters.is_single_bin_value(bin))
                bin = float(bin)
            for item in self.counters[counter_name]['bins']:
                if SecureCounters.is_in_bin(item[0], item[1], bin):
                    item[2] = ((int(item[2]) + int(inc))
                               % self.modulus)

    def _tally_counter(self, counter):
        if self.counters == None:
            return False

        # validate that the counter data structures match
        if not self._check_counter(counter):
            return False

        # ok, the counters match
        for key in self.counters:
            num_bins = len(self.counters[key]['bins'])
            for i in xrange(num_bins):
                tally_bin = self.counters[key]['bins'][i]
                tally_bin[2] = ((long(tally_bin[2]) +
                                 long(counter[key]['bins'][i][2]))
                                % self.modulus)

        # success
        return True

    def tally_counters(self, counters):
        # first add up all of the counters together
        for counter in counters:
            if not self._tally_counter(counter):
                return False
        # now adjust so our tally can register negative counts
        # (negative counts are possible if noise is negative)
        for key in self.counters:
            for tally_bin in self.counters[key]['bins']:
                tally_bin[2] = adjust_count_signed(tally_bin[2], self.modulus)
        return True

    def detach_counts(self):
        '''
        Asserts if we needed to add noise, and didn't add it
        '''
        assert not self.is_noise_pending
        counts = self.counters
        self.counters = None
        return counts


"""
def prob_exit(consensus_path, my_fingerprint, fingerprint_pool=None):
    '''
    this func is currently unused
    if it becomes used later, we must add stem as a required python library
    '''
    from stem.descriptor import parse_file

    if fingerprint_pool == None:
        fingerprint_pool = [my_fingerprint]

    net_status = next(parse_file(consensus_path, document_handler='DOCUMENT', validate=False))
    DW = float(net_status.bandwidth_weights['Wed'])/10000
    EW = float(net_status.bandwidth_weights['Wee'])/10000

    # we must use longs here, because otherwise sum_of_sq_bw can overflow on
    # platforms where python has 32-bit ints
    # (on these platforms, this happens when router_entry.bandwidth > 65535)
    my_bandwidth, DBW, EBW, sum_of_sq_bw = 0L, 0L, 0L, 0L

    if my_fingerprint in net_status.routers:
        my_bandwidth = net_status.routers[my_fingerprint].bandwidth

    for (fingerprint, router_entry) in net_status.routers.items():
        if fingerprint not in fingerprint_pool or 'BadExit' in router_entry.flags:
            continue

        if 'Guard' in router_entry.flags and 'Exit' in router_entry.flags:
            DBW += router_entry.bandwidth
            sum_of_sq_bw += router_entry.bandwidth**2

        elif 'Exit' in router_entry.flags:
            EBW += router_entry.bandwidth
            sum_of_sq_bw += router_entry.bandwidth**2

    TEWBW = DBW*DW + EBW*EW
    prob = my_bandwidth/TEWBW
    sum_of_sq = sum_of_sq_bw/(TEWBW**2)
    return prob, sum_of_sq
"""
