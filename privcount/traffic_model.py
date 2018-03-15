'''
Created on Dec 11, 2016

@author: rob

See LICENSE for licensing information
'''
import math
import logging

from json import loads

from privcount.counter import register_dynamic_counter, VITERBI_PACKETS_EVENT, VITERBI_STREAMS_EVENT, SecureCounters
SINGLE_BIN = SecureCounters.SINGLE_BIN

def float_value_is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def check_traffic_model_config(model_config):
    if model_config == None:
        logging.warning("Can't validate missing traffic model config")
        return False

    traffic_model_valid = True

    # we may have only one of stream model or packet model
    if 'packet_model' not in model_config and 'stream_model' not in model_config:
        logging.warning("Traffic model config does not contain key 'packet_model' or 'stream_model'")
        traffic_model_valid = False

    if 'packet_model' in model_config:
        packet_model = model_config['packet_model']
        if not __check_packet_model_config(packet_model):
            traffic_model_valid = False

    if 'stream_model' in model_config:
        stream_model = model_config['stream_model']
        if not __check_stream_model_config(stream_model):
            traffic_model_valid = False

    return traffic_model_valid

def __check_model_common_keys(model_config, model_type_str):
    for k in ['state_space', 'observation_space', 'emission_probability', 'transition_probability', 'start_probability']:
        if k not in model_config:
            logging.warning("{} model config does not contain required key '{}'".format(model_type_str, k))
            return False
    return True

def __check_model_common_start(model_config, model_type_str):
    valid = True
    if 'start_probability' in model_config:
        start_prob_sum = 0.0
        for state in model_config['start_probability']:
            if state not in model_config['state_space']:
                logging.warning("{} model config state space does not contain state '{}' which is used in the start probability array"
                    .format(model_type_str, state))
                valid = False
            start_prob_sum += float(model_config['start_probability'][state])
        if not float_value_is_close(start_prob_sum, 1.0):
            logging.warning("{} model config start probability sum '{}' does not equal 1.0 for state '{}'"
                .format(model_type_str, start_prob_sum, state))
            valid = False
    return valid

def __check_model_common_transition(model_config, model_type_str):
    valid = True
    if 'transition_probability' in model_config:
        for src_state in model_config['transition_probability']:
            if src_state not in model_config['state_space']:
                logging.warning("{} model config state space does not contain state '{}' which is used in the transition probability matrix"
                    .format(model_type_str, src_state))
                valid = False
            trans_prob_sum = 0.0
            for dst_state in model_config['transition_probability'][src_state]:
                if dst_state not in model_config['state_space']:
                    logging.warning("{} model config state space does not contain state '{}' which is used in the transition probability matrix"
                        .format(model_type_str, dst_state))
                    valid = False
                trans_prob_sum += float(model_config['transition_probability'][src_state][dst_state])

            if 'End' not in src_state and not float_value_is_close(trans_prob_sum, 1.0):
                logging.warning("{} model config transition probability sum '{}' does not equal 1.0 for src state '{}'"
                    .format(model_type_str, trans_prob_sum, src_state))
                valid = False
    return valid

def __check_model_common_emission(model_config, model_type_str):
    valid = True
    if 'emission_probability' in model_config:
        for state in model_config['emission_probability']:
            if state not in model_config['state_space']:
                logging.warning("{} model config state space does not contain state '{}' which is used in the emission probability matrix"
                    .format(model_type_str, state))
                valid = False
            emit_prob_sum = 0.0
            for obs in model_config['emission_probability'][state]:
                if obs not in model_config['observation_space']:
                    logging.warning("{} model config observation space does not contain state '{}' which is used in the emission probability matrix"
                        .format(model_type_str, obs))
                    valid = False
                if obs == 'F' and len(model_config['emission_probability'][state][obs]) != 1:
                    logging.warning("{} model config emission probability for state '{}' and observation '{}' does not contain required 1 param (dp)"
                        .format(model_type_str, state, obs))
                    valid = False
                elif (obs in ['$', '+', '-']) and len(model_config['emission_probability'][state][obs]) != 4:
                    logging.warning("{} model config emission probability for state '{}' and observation '{}' does not contain required 4 params (dp, mu, sigma, lambda)"
                        .format(model_type_str, state, obs))
                    valid = False
                elif obs not in ['$', '+', '-', 'F']:
                    logging.warning("{} model config observation '{}' in state '{}' does not match one of the recognized observation types (i.e., '+', '-', '$', or 'F')"
                        .format(model_type_str, obs, state))
                emit_prob_sum += float(model_config['emission_probability'][state][obs][0])
            if not float_value_is_close(emit_prob_sum, 1.0):
                logging.warning("{} model config emission probability sum '{}' does not equal 1.0 for state '{}'"
                    .format(model_type_str, emit_prob_sum, state))
                valid = False
    return valid

def __check_stream_model_config(model_config):
    '''
    A valid stream model config must contain the required keys for incrementing counters
    for this model, the start probabilities must sum to 1, and the transition probabilities
    for each outgoing state must sum to 1. This function returns True if the given
    model_config is valid, and False otherwise.
    '''
    if model_config == None:
        logging.warning("Can't validate missing stream model config")
        return False

    stream_model_valid = True

    if not __check_model_common_keys(model_config, "Stream"):
        stream_model_valid = False

    # packet model has its own observation types
    if 'observation_space' in model_config:
        obs_space = model_config['observation_space']
        if '$' not in obs_space or 'F' not in obs_space:
            logging.warning("Stream model config observation space is missing at least one of '$', and 'F'")
            stream_model_valid = False

    if not __check_model_common_start(model_config, "Stream"):
        stream_model_valid = False

    if not __check_model_common_transition(model_config, "Stream"):
        stream_model_valid = False

    if not __check_model_common_emission(model_config, "Stream"):
        stream_model_valid = False

    return stream_model_valid

def __check_packet_model_config(model_config):
    '''
    A valid packet model config must contain the required keys for incrementing counters
    for this model, the start probabilities must sum to 1, and the transition probabilities
    for each outgoing state must sum to 1. This function returns True if the given
    model_config is valid, and False otherwise.
    '''
    if model_config == None:
        logging.warning("Can't validate missing packet model config")
        return False

    packet_model_valid = True

    if not __check_model_common_keys(model_config, "Packet"):
        packet_model_valid = False

    # packet model has its own observation types
    if 'observation_space' in model_config:
        obs_space = model_config['observation_space']
        if '+' not in obs_space or '-' not in obs_space or 'F' not in obs_space:
            logging.warning("Packet model config observation space is missing at least one of '+', '-', and 'F'")
            packet_model_valid = False

    if not __check_model_common_start(model_config, "Packet"):
        packet_model_valid = False

    if not __check_model_common_transition(model_config, "Packet"):
        packet_model_valid = False

    if not __check_model_common_emission(model_config, "Packet"):
        packet_model_valid = False

    return packet_model_valid

def __normalize_traffic_model_config_helper(model_config):
    '''
    Normalizes the given packet or stream model config.
    '''

    if 'start_probability' in model_config:
        start_prob_sum = sum(model_config['start_probability'].values())
        for state in model_config['start_probability']:
            model_config['start_probability'][state] /= start_prob_sum

    if 'emission_probability' in model_config:
        for state in model_config['emission_probability']:
            emit_prob_sum = 0.0
            for obs in model_config['emission_probability'][state]:
                if len(model_config['emission_probability'][state][obs]) > 0:
                    emit_prob_sum += float(model_config['emission_probability'][state][obs][0])
            for obs in model_config['emission_probability'][state]:
                if len(model_config['emission_probability'][state][obs]) > 0:
                    model_config['emission_probability'][state][obs][0] /= emit_prob_sum

    if 'transition_probability' in model_config:
        for src in model_config['transition_probability']:
            trans_prob_sum = sum(model_config['transition_probability'][src].values())
            for dst in model_config['transition_probability'][src]:
                model_config['transition_probability'][src][dst] /= trans_prob_sum

    return model_config

def normalize_traffic_model_config(model_config):
    '''
    Normalizes the given traffic model config so that all start,
    emission, and transition probabilities sum to 1.0. Returns
    the normalized config.
    '''
    # we may have only one of stream model or packet model
    if 'packet_model' in model_config:
        model_config['packet_model'] = __normalize_traffic_model_config_helper(model_config['packet_model'])
    if 'stream_model' in model_config:
        model_config['stream_model'] = __normalize_traffic_model_config_helper(model_config['stream_model'])
    return model_config

class TrafficModel(object):
    '''
    A class that represents a traffic model, i.e., a combination
    of packet and stream hidden markov models.
    See `test/traffic.model.json` for a simple traffic model that this class can represent.
    '''

    def __init__(self, model_config):
        '''
        Initialize the model with a set of states, probabilities for starting in each of those
        states, probabilities for transitioning between those states, and proababilities of emitting
        certain types of events in each of those states.

        For us, the states very loosely represent if the node is transmitting or pausing.
        The events represent if we saw an outbound or inbound packet while in each of those states.
        '''
        if not check_traffic_model_config(model_config):
            return None

        self.packet_hmm = None
        if 'packet_model' in model_config:
            # the packet model counters are counted on stream end events
            self.packet_hmm = HiddenMarkovModel(model_config['packet_model'], "ExitStreamTrafficModel")

        self.stream_hmm = None
        if 'stream_model' in model_config:
            # the stream model counters are counted on circuit end events
            self.stream_hmm = HiddenMarkovModel(model_config['stream_model'], "ExitCircuitTrafficModel")

    def register_counters(self):
        if self.packet_hmm != None:
            p = self.packet_hmm.get_dynamic_counter_template_label_mapping()
            for label in p:
                register_dynamic_counter(label, { VITERBI_PACKETS_EVENT })

        if self.stream_hmm != None:
            s = self.stream_hmm.get_dynamic_counter_template_label_mapping()
            for label in s:
                register_dynamic_counter(label, { VITERBI_STREAMS_EVENT })

    def __get_dynamic_counter_template_label_mapping(self):
        d = {}
        # we may have only one of stream model or packet model
        if self.packet_hmm != None:
            p = self.packet_hmm.get_dynamic_counter_template_label_mapping()
            for k in p: d[k] = p[k]
        if self.stream_hmm != None:
            s = self.stream_hmm.get_dynamic_counter_template_label_mapping()
            for k in s: d[k] = s[k]
        return d

    def __get_static_counter_template_label_mapping(self):
        d = {}
        # we may have only one of stream model or packet model
        if self.packet_hmm != None:
            p = self.packet_hmm.get_static_counter_template_label_mapping()
            for k in p: d[k] = p[k]
        if self.stream_hmm != None:
            s = self.stream_hmm.get_static_counter_template_label_mapping()
            for k in s: d[k] = s[k]
        return d

    def __get_all_counter_template_label_mapping(self):
        '''
        Get a dict mapping all static and dynamic counter labels for this model to the
        template label that is used to specify noise for this model.
        '''
        all_labels = self.__get_dynamic_counter_template_label_mapping()
        static_labels = self.__get_static_counter_template_label_mapping()
        for label in static_labels:
            all_labels[label] = static_labels[label]
        return all_labels

    def __get_all_template_labels(self):
        '''
        Return the set of all template labels that are used to specify noise for this model.
        '''
        all_labels = self.__get_all_counter_template_label_mapping()
        return set(all_labels.values())

    def __get_dynamic_counter_labels(self):
        '''
        Return the set of counters that should be counted for this model,
        but only those whose name depends on the traffic model input.
        '''
        dynamic_labels = self.__get_dynamic_counter_template_label_mapping()
        return set(dynamic_labels.keys())

    def get_all_counter_labels(self):
        '''
        Return the set of all counters that will be counted for this model.
        '''
        all_labels = self.__get_all_counter_template_label_mapping()
        return set(all_labels.keys())

    def check_noise_config(self, templated_noise_config):
        '''
        Return True if the given templated_noise_config contains the required keys
        for specifying noise for this model, False otherwise.
        '''
        if 'counters' not in templated_noise_config:
            logging.warning("'counters' not in noise config. Please check config file.")
            return False

        traffic_noise_valid = True
        for key in templated_noise_config['counters']:
            if key not in self.__get_all_template_labels():
                logging.warning("Could not find key '{}' in set of acceptable labels.".format(key))
                traffic_noise_valid = False
        return traffic_noise_valid

    def get_noise_init_config(self, templated_noise_config):
        '''
        Expand the templated noise config into a full set of noise params for all of the counters
        counted by this model. Return a dict with the proper initial noise parameters for all
        labels, given the noise values specified in templated_noise_config. Returns None if the
        noise config is invalid.
        '''
        if not self.check_noise_config(templated_noise_config):
            return None

        noise_dict = {}

        label_map = self.__get_all_counter_template_label_mapping()
        for counter_label in label_map:
            template_label = label_map[counter_label]
            # only count the label if it was in the config
            if template_label in templated_noise_config['counters']:
                noise_dict[counter_label] = templated_noise_config['counters'][template_label]

        return noise_dict

    def get_bins_init_config(self, templated_noise_config):
        '''
        Returns the initial bin values for all counters counted by this model.
        If templated_noise_config is None, collect all counters. Otherwise,
        collect those specified in templated_noise_config.
        '''
        bins_dict = {}

        label_map = self.__get_all_counter_template_label_mapping()
        for counter_label in label_map:
            template_label = label_map[counter_label]
            # only count the label if it was in the config
            if template_label in templated_noise_config['counters']:
                # traffic counters are all single bin counters
                bins_dict[counter_label] = {'bins': [[0.0, float("inf")]]}

        return bins_dict

    def increment_packets_counters(self, viterbi_result, secure_counters):
        '''
        Increment the appropriate secure counter labels for this model,
        based on the viterbi path for packets sent on this stream.
        '''
        if self.packet_hmm != None:
            # the fact that we got an event means that we observed a stream
            secure_counters.increment('ExitStreamTrafficModelStreamCount',
                                      bin=SINGLE_BIN,
                                      inc=1)
            self.packet_hmm.increment_counters(viterbi_result, secure_counters)

    def increment_streams_counters(self, viterbi_result, secure_counters):
        '''
        Increment the appropriate secure counter labels for this model,
        based on the viterbi path for packets sent on this stream.
        '''
        if self.stream_hmm != None:
            # the fact that we got an event means that we observed a stream
            secure_counters.increment('ExitCircuitTrafficModelCircuitCount',
                                      bin=SINGLE_BIN,
                                      inc=1)
            self.stream_hmm.increment_counters(viterbi_result, secure_counters)

    def update_from_tallies(self, tallies, trans_inertia=0.5, emit_inertia=0.5):
        '''
        Given the (noisy) aggregated tallied counter values for
        this model, compute the updated packet and stream models.
        '''

        updated_model_config = {}

        if self.packet_hmm != None:
            updated_model_config['packet_model'] = self.packet_hmm.update_from_tallies(tallies, trans_inertia, emit_inertia)

        if self.stream_hmm != None:
            updated_model_config['stream_model'] = self.stream_hmm.update_from_tallies(tallies, trans_inertia, emit_inertia)

        return updated_model_config

class HiddenMarkovModel(object):
    '''
    A private class that represents a hidden markov model (HMM).
    '''

    def __init__(self, model_config, counter_prefix_str):
        '''
        Initialize the model with a set of states, probabilities for starting in each of those
        states, probabilities for transitioning between those states, and proababilities of emitting
        certain types of events in each of those states.

        For us, the states very loosely represent if the node is transmitting or pausing.
        The events represent if we saw an outbound or inbound packet, or a stream, while in each of those states.
        '''

        self.state_s = model_config['state_space']
        self.obs_s = model_config['observation_space']
        self.start_p = model_config['start_probability']
        self.trans_p = model_config['transition_probability']
        self.emit_p = model_config['emission_probability']
        self.prefix = counter_prefix_str

    def get_dynamic_counter_template_label_mapping(self):
        '''
        Get a dict mapping the dynamic counter labels for this model to the template label
        that is used to specify noise for this model. Dynamic counter labels are
        dependent on the model input.

        We count the following, for all states and packet directions:
          + the total number of emissions
          + the total number of transitions
          + for packet events ('+' or '-'), and stream events ('$') for dwell state
            + the sum of log delays between events
            + the sum of squared log delays between events (to compute the variance)
          + for stream events ('$') for active state
            + the sum of delays (along with num emissions to compute new rate)
        '''
        labels = {}

        for state in self.emit_p:
            for obs in self.emit_p[state]:
                template_label = "{}EmissionCount_<STATE>_<OBS>".format(self.prefix)
                counter_label = "{}EmissionCount_{}_{}".format(self.prefix, state, obs)
                labels[counter_label] = template_label

                if obs == '+' or obs == '-' or (obs == '$' and 'Dwell' in state):
                    # counters to update a lognormal distribution
                    template_label = "{}LogDelayTime_<STATE>_<OBS>".format(self.prefix)
                    counter_label = "{}LogDelayTime_{}_{}".format(self.prefix, state, obs)
                    labels[counter_label] = template_label

                    template_label = "{}SquaredLogDelayTime_<STATE>_<OBS>".format(self.prefix)
                    counter_label = "{}SquaredLogDelayTime_{}_{}".format(self.prefix, state, obs)
                    labels[counter_label] = template_label
                elif obs == '$' and 'Active' in state:
                    # counters to update an exponential distribution
                    template_label = "{}DelayTime_<STATE>_<OBS>".format(self.prefix)
                    counter_label = "{}DelayTime_{}_{}".format(self.prefix, state, obs)
                    labels[counter_label] = template_label

        for src_state in self.trans_p:
            for dst_state in self.trans_p[src_state]:
                if self.trans_p[src_state][dst_state] > 0.0:
                    template_label = "{}TransitionCount_<SRCSTATE>_<DSTSTATE>".format(self.prefix)
                    counter_label = "{}TransitionCount_{}_{}".format(self.prefix, src_state, dst_state)
                    labels[counter_label] = template_label

        for state in self.start_p:
            if self.start_p[state] > 0.0:
                template_label = "{}TransitionCount_START_<STATE>".format(self.prefix)
                counter_label = "{}TransitionCount_START_{}".format(self.prefix, state)
                labels[counter_label] = template_label

        return labels

    def get_static_counter_template_label_mapping(self):
        '''
        Get a dict mapping the static counter labels for this model to the template label
        that is used to specify noise for this model. Static counter labels are not
        dependent on the model input.

        ExitStreamTrafficModelDelayTime
        ExitStreamTrafficModelEmissionCount
        ExitStreamTrafficModelLogDelayTime
        ExitStreamTrafficModelSquaredLogDelayTime
        ExitStreamTrafficModelStreamCount
        ExitStreamTrafficModelTransitionCount

        ExitCircuitTrafficModelDelayTime
        ExitCircuitTrafficModelEmissionCount
        ExitCircuitTrafficModelLogDelayTime
        ExitCircuitTrafficModelSquaredLogDelayTime
        ExitCircuitTrafficModelStreamCount
        ExitCircuitTrafficModelTransitionCount
        '''
        static_labels = ['{}StreamCount'.format(self.prefix),
                         '{}EmissionCount'.format(self.prefix),
                         '{}TransitionCount'.format(self.prefix),
                         '{}DelayTime'.format(self.prefix),
                         '{}LogDelayTime'.format(self.prefix),
                         '{}SquaredLogDelayTime'.format(self.prefix)]
        labels = {}
        for static_label in static_labels:
            labels[static_label] = static_label
        return labels

    def increment_counters(self, viterbi_result, secure_counters):
        '''
        Increment the appropriate secure counter labels for this model given the observed
        list of events specifying when bytes were transferred in Tor.
          viterbi_result: the viterbi path through our model given the observed delays
          secure_counters: the SecureCounters object whose counters should get incremented
            as a result of the observed bytes events

        The viterbi_result is encoded as a json string, e.g.:
          '[["m10s1";"+";35432];["m2s4";"+";0];["m4s2";"-";100];["m4sEnd";"F";0]]'

        This is a list of observations, where each observation has a state name,
        an observation code, and a delay value.
        '''

        # python lets you encode with non-default separators, but not decode
        viterbi_result = viterbi_result.replace(';',',')
        path = loads(viterbi_result)

        # empty lists are possible, when there was in error in the Tor
        # viterbi code, or when a stream ended with no data sent.
        # if we have an empty list, the following loop will not execute.
        for i, packet in enumerate(path):
            if len(packet) < 3:
                continue

            state, obs, delay = str(packet[0]), str(packet[1]), int(packet[2])

            # delay of 0 indicates the packets were observed at the same time
            # log(x=0) is undefined, and log(x<1) is negative
            # we don't want to count negatives, so override delay if needed
            ldelay = 0 if delay < 1 else int(round(math.log(delay)))

            secure_counters.increment('{}EmissionCount'.format(self.prefix),
                                      bin=SINGLE_BIN,
                                      inc=1)
            label = '{}EmissionCount_{}_{}'.format(self.prefix, state, obs)
            secure_counters.increment(label,
                                      bin=SINGLE_BIN,
                                      inc=1)

            if obs == '+' or obs == '-' or (obs == '$' and 'Dwell' in state):
                secure_counters.increment('{}LogDelayTime'.format(self.prefix),
                                          bin=SINGLE_BIN,
                                          inc=ldelay)
                label = '{}LogDelayTime_{}_{}'.format(self.prefix, state, obs)
                secure_counters.increment(label,
                                          bin=SINGLE_BIN,
                                          inc=ldelay)

                secure_counters.increment('{}SquaredLogDelayTime'.format(self.prefix),
                                          bin=SINGLE_BIN,
                                          inc=ldelay*ldelay)
                label = '{}SquaredLogDelayTime_{}_{}'.format(self.prefix, state, obs)
                secure_counters.increment(label,
                                          bin=SINGLE_BIN,
                                          inc=ldelay*ldelay)
            elif obs == '$' and 'Active' in state:
                secure_counters.increment('{}DelayTime'.format(self.prefix),
                                          bin=SINGLE_BIN,
                                          inc=delay)
                label = '{}DelayTime_{}_{}'.format(self.prefix, state, obs)
                secure_counters.increment(label,
                                          bin=SINGLE_BIN,
                                          inc=delay)

            if i == 0: # track starting transitions
                label = '{}TransitionCount_START_{}'.format(self.prefix, state)
                secure_counters.increment(label,
                                          bin=SINGLE_BIN,
                                          inc=1)

            # track transitions for all but the final state
            if (i + 1) < len(path) and len(path[i + 1]) >= 3:
                next_state = str(path[i + 1][0])
                secure_counters.increment('{}TransitionCount'.format(self.prefix),
                                          bin=SINGLE_BIN,
                                          inc=1)
                label = '{}TransitionCount_{}_{}'.format(self.prefix, state, next_state)
                secure_counters.increment(label,
                                          bin=SINGLE_BIN,
                                          inc=1)

    def __normalize_start_tallies(self, tallies):
        '''
        Converts the given tally counts to valid probabilities by normalizing the counts.
        Returns start probability dictionary whose probability sum is guaranteed to sum to
        1.0 (i.e., 100%), or 0.0 if the we counted no positive start counts.
        '''

        # handle start probabilities.
        s_count = {}

        # first we go through and convert tallies to valid non-negative values
        for state in self.start_p:
            label = "{}TransitionCount_START_{}".format(self.prefix, state)
            count = tallies[label] if label in tallies and tallies[label] > 0.0 else 0.0
            s_count[state] = float(count)

        # normalize counts to ensure the probabilities sum to 100%,
        # or set to 0 if we had absolutely no positive counts.
        s_total = float(sum(s_count.values()))
        for state in s_count:
            if s_total > 0.0:
                s_count[state] = s_count[state]/s_total
            else:
                s_count[state] = 0.0

        return s_count

    def __update_start_probabilities(self, s_tally_prob, inertia):
        '''
        Update the start probabilities based on the normalized probabilities from the
        tally count results.
        Updates self.start_p in place and also returns a reference to it.
        '''

        # apply inertia
        for state in self.start_p:
            new_prob = inertia * self.start_p[state] + (1.0-inertia) * s_tally_prob[state]
            self.start_p[state] = new_prob

        # be extra sure our new probs sum to 1.0
        s_total = float(sum(self.start_p.values()))
        if s_total > 0.0:
            for state in self.start_p:
                self.start_p[state] = self.start_p[state]/s_total
        else:
            logging.warning("BUG: we have no positive start probabilities in updated model.")

        return self.start_p


    def __normalize_transition_tallies(self, tallies):
        '''
        Converts the given tally counts to valid probabilities by normalizing the counts.
        Returns transition probability dictionary whose probability sums are guaranteed to sum to
        1.0 (i.e., 100%), or 0.0 if the we counted no positive start counts.
        '''

        # handle transition probabilities.
        t_count = {}

        for src_state in self.trans_p:
            # first we go through and convert tallies to valid non-negative values
            t_count[src_state] = {}
            for dst_state in self.trans_p[src_state]:
                label = "{}TransitionCount_{}_{}".format(self.prefix, src_state, dst_state)
                count = tallies[label] if label in tallies and tallies[label] > 0.0 else 0.0
                t_count[src_state][dst_state] = float(count)

            # normalize counts to ensure the probabilities sum to 100%,
            # or set to 0 if we had absolutely no positive counts.
            t_total = float(sum(t_count[src_state].values()))
            for dst_state in t_count[src_state]:
                if t_total > 0.0:
                    t_count[src_state][dst_state] = t_count[src_state][dst_state]/t_total
                else:
                    t_count[src_state][dst_state] = 0.0

        return t_count

    def __update_transition_probabilities(self, t_tally_prob, inertia):
        '''
        Update the transition probabilities based on the normalized probabilities from the
        tally count results.
        Updates self.trans_p in place and also returns a reference to it.
        '''

        for src_state in self.trans_p:
            # apply inertia
            for dst_state in self.trans_p[src_state]:
                new_prob = inertia * self.trans_p[src_state][dst_state] + (1.0-inertia) * t_tally_prob[src_state][dst_state]
                self.trans_p[src_state][dst_state] = new_prob

            # be extra sure our new probs sum to 1.0
            t_total = float(sum(self.trans_p[src_state].values()))
            if t_total > 0.0:
                for dst_state in self.trans_p[src_state]:
                    self.trans_p[src_state][dst_state] = self.trans_p[src_state][dst_state]/t_total
            else:
                if 'End' in src_state:
                    logging.info("No positive transition probabilities for end state {}. This is OK."
                        .format(src_state))
                else:
                    logging.warning("BUG: we have no positive transition probabilities in updated model for state {}."
                        .format(src_state))

        return self.trans_p

    def __normalize_emission_tallies(self, tallies):
        '''
        Converts the given tally counts to valid probabilities by normalizing the counts.
        Returns emission probability dictionary whose probability sums are guaranteed to sum to
        1.0 (i.e., 100%), or 0.0 if the we counted no positive start counts.
        '''

        e_count, e_mu, e_sigma, e_lambda = {}, {}, {}, {}
        for state in self.emit_p:
            # first we go through and convert tallies to valid non-negative values
            e_count[state], e_mu[state], e_sigma[state], e_lambda[state] = {}, {}, {}, {}
            for obs in self.emit_p[state]:
                sd_label = "{}EmissionCount_{}_{}".format(self.prefix, state, obs)
                if sd_label in tallies and tallies[sd_label] > 0.0:
                    e_count[state][obs] = float(tallies[sd_label])
                else:
                    e_count[state][obs] = 0.0

                if obs == '+' or obs == '-' or (obs == '$' and 'Dwell' in state):
                    mu_label = "{}LogDelayTime_{}_{}".format(self.prefix, state, obs)
                    if mu_label in tallies and tallies[mu_label] > 0.0 and e_count[state][obs] > 0.0:
                        e_mu[state][obs] = float(tallies[mu_label])/float(e_count[state][obs])
                    else:
                        e_mu[state][obs] = 0.0

                    ss_label = "{}SquaredLogDelayTime_{}_{}".format(self.prefix, state, obs)
                    if ss_label in tallies and tallies[ss_label] > 0.0 and sd_label in tallies and tallies[sd_label] > 0.0:
                        obs_var = float(tallies[ss_label])/float(tallies[sd_label])
                        obs_var -= e_mu[state][obs]**2
                    else:
                        obs_var = 0.0

                    # rounding errors or noise can make a small positive variance look negative
                    # setting a small "sane default" for this case
                    if obs_var < math.sqrt(0.01):
                        e_sigma[state][obs] = 0.01
                    else: # No rounding errors, do the math
                        e_sigma[state][obs] = math.sqrt(obs_var)
                elif obs == '$' and 'Active' in state:
                    lam_label = "{}DelayTime_{}_{}".format(self.prefix, state, obs)
                    if lam_label in tallies and tallies[lam_label] > 0.0 and e_count[state][obs] > 0.0:
                        # the rate is streams per microsecond
                        # e_count is the total emissions (streams), tallies is the total delay (microsecs)
                        e_lambda[state][obs] = float(e_count[state][obs])/float(tallies[lam_label])
                    else:
                        e_lambda[state][obs] = 0.0

            # normalize counts to ensure the probabilities sum to 100%,
            # or set to 0 if we had absolutely no positive counts.
            e_total = float(sum(e_count[state].values()))
            for obs in e_count[state]:
                if e_total > 0.0:
                    e_count[state][obs] = e_count[state][obs]/e_total
                else:
                    e_count[state][obs] = 0.0

        return e_count, e_mu, e_sigma, e_lambda

    def __update_emission_probabilities(self, e_count, e_mu, e_sigma, e_lambda, inertia):
        '''
        Update the emission probabilities based on the normalized probabilities from the
        tally count results.
        Updates self.emit_p in place and also returns a reference to it.
        '''

        for state in self.emit_p:
            # apply inertia
            for obs in self.emit_p[state]:
                if obs == '+' or obs == '-' or (obs == '$' and 'Dwell' in state):
                    (dp, mu, sigma, lam) = self.emit_p[state][obs]

                    dp_new = inertia * dp + (1.0-inertia) * e_count[state][obs]
                    mu_new = inertia * mu + (1.0-inertia) * e_mu[state][obs]
                    sigma_new = inertia * sigma + (1.0-inertia) * e_sigma[state][obs]

                    self.emit_p[state][obs] = (dp_new, mu_new, sigma_new, 0.0)
                elif obs == '$' and 'Active' in state:
                    (dp, mu, sigma, lam) = self.emit_p[state][obs]

                    dp_new = inertia * dp + (1.0-inertia) * e_count[state][obs]
                    lam_new = inertia * lam + (1.0-inertia) * e_lambda[state][obs]

                    self.emit_p[state][obs] = (dp_new, 0.0, 0.0, lam_new)

            # be extra sure our new probs sum to 1.0
            e_total = float(sum([v[0] for v in self.emit_p[state].values()]))
            if e_total > 0.0:
                for obs in self.emit_p[state]:
                    if obs == '+' or obs == '-' or (obs == '$' and 'Dwell' in state):
                        (dp, mu, sigma, lam) = self.emit_p[state][obs]
                        dp_new = dp/e_total
                        self.emit_p[state][obs] = (dp_new, mu, sigma, 0.0)
                    elif obs == '$' and 'Active' in state:
                        (dp, mu, sigma, lam) = self.emit_p[state][obs]
                        dp_new = dp/e_total
                        self.emit_p[state][obs] = (dp_new, 0.0, 0.0, lam)
            else:
                logging.warning("BUG: we have no positive emission probabilities in updated model for state {}."
                    .format(state))

        return self.emit_p

    def update_from_tallies(self, tallies, trans_inertia, emit_inertia):
        '''
        + Transition probabilities - trans_p[s][t] = inertia*trans_p[s][t] + (1-inertia)*(tally result)
        + Emission probabilities - emit_p[s][d] <- dp', mu', sigma' where
          dp' <- emit_inertia * dp + (1-emit_inertia)*(state_direction count / state count)
          mu' <- emit_inertia * mu + (1-emit_inertia)*(log-delay sum / state_direction count)
          sigma' <- emit_inertia * sigma + (1-emit_inertia)*sqrt(avg. of squares - square of average)
        '''

        # handle start probabilities
        s_tally_prob = self.__normalize_start_tallies(tallies)
        self.start_p = self.__update_start_probabilities(s_tally_prob, trans_inertia)

        # handle transition probabilities
        t_tally_prob = self.__normalize_transition_tallies(tallies)
        self.trans_p = self.__update_transition_probabilities(t_tally_prob, trans_inertia)

        # handle emission probabilities
        e_count, e_mu, e_sigma, e_lambda = self.__normalize_emission_tallies(tallies)
        self.emit_p = self.__update_emission_probabilities(e_count, e_mu, e_sigma, e_lambda, emit_inertia)

        updated_model_config = {
            'state_space': self.state_s,
            'observation_space': self.obs_s,
            'start_probability': self.start_p,
            'transition_probability': self.trans_p,
            'emission_probability': self.emit_p
        }

        return updated_model_config
