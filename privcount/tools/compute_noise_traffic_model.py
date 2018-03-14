# See LICENSE for licensing information

import math
import yaml

import privcount.statistics_noise as psn

## useful constants
NUM_SECONDS_PER_DAY = 24*60*60
NUM_MICROSECONDS_PER_SECOND = 1000*1000
NUM_MICROSECONDS_PER_DAY = NUM_SECONDS_PER_DAY * NUM_MICROSECONDS_PER_SECOND

## measurement parameters ##
epoch_length_days = 4.0 # length of collection round in days
epoch_length_seconds = epoch_length_days * NUM_SECONDS_PER_DAY
epoch_length_microseconds = epoch_length_days * NUM_MICROSECONDS_PER_DAY
slice_length_seconds = 10*60 # time slice for sensitive data in seconds
slice_length_microseconds = slice_length_seconds * NUM_MICROSECONDS_PER_SECOND
num_relay_machines = 3 # number of relays collecting statistics

## privacy sensitivity ##
sensitivity_circuits = 6*24+2 # constant use for 24 hours: two pre-emptive circuits plus six circuits per hour (due to a circuit lifetime of 10 minutes)
sensitivity_streams = 150 * 200 # max of likely applications (viz. Web)
sensitivity_kibytes = 10*1024 # 10 MiB, should cover nearly all web pages (see total transfer size per page distribution at <http://httparchive.org/interesting.php>)

sensitivity_packets = sensitivity_kibytes * 1024.0 / 1500.0
sensitivity_packet_logdelay = sensitivity_circuits * min(slice_length_microseconds, epoch_length_microseconds) * math.log(2)/2.0
sensitivity_packet_logdelaysquared = 1 # TODO this value is temporary

## estimated values ##
# taken from initial data collection on 4/29/16
num_kibytes_per_day = 2207373238.0
num_packets_per_day = num_kibytes_per_day * 1024.0 / 1500.0
# taken from p2p initial data collection 6/16/16-6/17/16
num_streams_per_day = 58157663.0/2
num_circuits_per_day = 8131540.0/2
# guessed values - these assume all packets are equally spaced
logdelay_per_packet = math.log(NUM_MICROSECONDS_PER_DAY/num_packets_per_day)
num_packet_logdelay_per_day = logdelay_per_packet * num_packets_per_day
num_packet_logdelaysquared_per_day = logdelay_per_packet * logdelay_per_packet * num_packets_per_day

# parameters that will be re-used
stream_single_parameters = (sensitivity_streams, num_streams_per_day * epoch_length_days)
stream_histogram_parameters = (2 * sensitivity_streams, num_streams_per_day * epoch_length_days)
packet_single_parameters = (sensitivity_packets, num_packets_per_day * epoch_length_days)
packet_histogram_parameters = (2 * sensitivity_packets, num_packets_per_day * epoch_length_days)
packet_logdelay_single_parameters = (sensitivity_packet_logdelay, num_packet_logdelay_per_day * epoch_length_days)
packet_logdelay_histogram_parameters = (2 * sensitivity_packet_logdelay, num_packet_logdelay_per_day * epoch_length_days)
packet_logdelaysquared_single_parameters = (sensitivity_packet_logdelaysquared, num_packet_logdelaysquared_per_day * epoch_length_days)
packet_logdelaysquared_histogram_parameters = (2 * sensitivity_packet_logdelaysquared, num_packet_logdelaysquared_per_day * epoch_length_days)

traffic_model_parameters = {
    # "single counter" type of statistics
    'ExitStreamTrafficModelStreamCount': stream_single_parameters,
    'ExitStreamTrafficModelEmissionCount': packet_single_parameters,
    'ExitStreamTrafficModelLogDelayTime': packet_logdelay_single_parameters,
    'ExitStreamTrafficModelSquaredLogDelayTime': packet_logdelaysquared_single_parameters,
    'ExitStreamTrafficModelTransitionCount': packet_single_parameters,

    # "histogram" type of statistics
    # each '<>' is expanded to include all counters for the states defined in a particular model
    'ExitStreamTrafficModelEmissionCount_<STATE>_<OBS>': packet_histogram_parameters,
    'ExitStreamTrafficModelLogDelayTime_<STATE>_<OBS>': packet_logdelay_histogram_parameters,
    'ExitStreamTrafficModelSquaredLogDelayTime_<STATE>_<OBS>': packet_logdelaysquared_histogram_parameters,
    # only counted on stream's first packet, so use stream params
    'ExitStreamTrafficModelTransitionCount_START_<STATE>': stream_histogram_parameters,
    # counted on all but the last packet on each stream
    'ExitStreamTrafficModelTransitionCount_<SRCSTATE>_<DSTSTATE>': packet_histogram_parameters,
}

if __name__ == '__main__':
    epsilon = 0.3
    delta = 1e-3
    excess_noise_ratio = num_relay_machines # factor by which noise is expanded to allow for malicious relays
    sigma_tol = psn.DEFAULT_SIGMA_TOLERANCE
    epsilon_tol = psn.DEFAULT_EPSILON_TOLERANCE
    sigma_ratio_tol = psn.DEFAULT_SIGMA_RATIO_TOLERANCE

    # get optimal noise allocation for initial statistics
    (epsilons, sigmas, sigma_ratio) =  psn.get_opt_privacy_allocation(epsilon,
        delta, traffic_model_parameters, excess_noise_ratio, sigma_tol=sigma_tol,
        epsilon_tol=epsilon_tol, sigma_ratio_tol=sigma_ratio_tol)

    # print information about traffic model statistics noise
    print('\n* Traffic model statistics *\n')

    psn.print_privacy_allocation(traffic_model_parameters, sigmas, epsilons, excess_noise_ratio)

    psn.compare_noise_allocation(epsilon, delta, traffic_model_parameters,
                             excess_noise_ratio,
                             sigma_tol=sigma_tol,
                             epsilon_tol=epsilon_tol,
                             sigma_ratio_tol=sigma_ratio_tol,
                             sanity_check=psn.DEFAULT_DUMMY_COUNTER_NAME)

    noise_parameters =\
        psn.get_noise_allocation_stats(epsilon, delta, traffic_model_parameters,
                                   excess_noise_ratio,
                                   sigma_tol=sigma_tol,
                                   epsilon_tol=epsilon_tol,
                                   sigma_ratio_tol=sigma_ratio_tol,
                                   sanity_check=psn.DEFAULT_DUMMY_COUNTER_NAME)
    print('Traffic model noise config\n-----')
    print yaml.dump(noise_parameters, default_flow_style=False)
