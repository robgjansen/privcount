'''
Created on Dec 11, 2016

@author: rob

See LICENSE for licensing information
'''
import math
import logging

from json import loads

from privcount.counter import register_dynamic_counter, VITERBI_EVENT, SecureCounters
SINGLE_BIN = SecureCounters.SINGLE_BIN

def float_value_is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def check_traffic_model_config(model_config):
    '''
    A valid traffic model config mmust contain the required keys for incrementing counters
    for this model, the start probabilities must sum to 1, and the transition probabilities
    for each outgoing state must sum to 1. This function returns True if the given
    model_config is valid, and False otherwise.
    '''
    if model_config == None:
        return False

    traffic_model_valid = True

    for k in ['state_space', 'observation_space', 'emission_probability', 'transition_probability', 'start_probability']:
        if k not in model_config:
            logging.warning("Traffic model config does not contain required key '{}'".format(k))
            traffic_model_valid = False

    if 'observation_space' in model_config:
        obs_space = model_config['observation_space']
        if '+' not in obs_space or '-' not in obs_space or 'F' not in obs_space:
            logging.warning("Traffic model config observation space is missing at least one of '+', '-', and 'F'")
            traffic_model_valid = False

    if 'start_probability' in model_config:
        start_prob_sum = 0.0
        for state in model_config['start_probability']:
            if state not in model_config['state_space']:
                logging.warning("Traffic model config state space does not contain state '{}' which is used in the start probability array"
                    .format(state))
                traffic_model_valid = False
            start_prob_sum += float(model_config['start_probability'][state])
        if not float_value_is_close(start_prob_sum, 1.0):
            logging.warning("Traffic model config start probability sum '{}' does not equal 1.0 for state '{}'"
                .format(start_prob_sum, state))
            traffic_model_valid = False

    if 'transition_probability' in model_config:
        for src_state in model_config['transition_probability']:
            if src_state not in model_config['state_space']:
                logging.warning("Traffic model config state space does not contain state '{}' which is used in the transition probability matrix"
                    .format(src_state))
                traffic_model_valid = False
            trans_prob_sum = 0.0
            for dst_state in model_config['transition_probability'][src_state]:
                if dst_state not in model_config['state_space']:
                    logging.warning("Traffic model config state space does not contain state '{}' which is used in the transition probability matrix"
                        .format(dst_state))
                    traffic_model_valid = False
                trans_prob_sum += float(model_config['transition_probability'][src_state][dst_state])

            if 'End' not in src_state and not float_value_is_close(trans_prob_sum, 1.0):
                logging.warning("Traffic model config transition probability sum '{}' does not equal 1.0 for src state '{}'"
                    .format(trans_prob_sum, src_state))
                traffic_model_valid = False

    if 'emission_probability' in model_config:
        for state in model_config['emission_probability']:
            if state not in model_config['state_space']:
                logging.warning("Traffic model config state space does not contain state '{}' which is used in the emission probability matrix"
                    .format(state))
                traffic_model_valid = False
            emit_prob_sum = 0.0
            for obs in model_config['emission_probability'][state]:
                if obs not in model_config['observation_space']:
                    logging.warning("Traffic model config observation space does not contain state '{}' which is used in the emission probability matrix"
                        .format(obs))
                    traffic_model_valid = False
                if len(model_config['emission_probability'][state][obs]) != 3:
                    logging.warning("Traffic model config emission probability for state '{}' and observation '{}' does not contain required 3 params (dp, mu, sigam)"
                        .format(state, obs))
                    traffic_model_valid = False
                else:
                    emit_prob_sum += float(model_config['emission_probability'][state][obs][0])
            if not float_value_is_close(emit_prob_sum, 1.0):
                logging.warning("Traffic model config emission probability sum '{}' does not equal 1.0 for state '{}'"
                    .format(emit_prob_sum, state))
                traffic_model_valid = False

    return traffic_model_valid

def normalize_traffic_model_config(model_config):
    '''
    Normalizes the given traffic model config so that all start,
    emission, and transition probabilities sum to 1.0. Returns
    the normalized config.
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

class TrafficModel(object):
    '''
    A class that represents a hidden markov model (HMM).
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

        self.config = model_config
        self.state_s = self.config['state_space']
        self.obs_s = self.config['observation_space']
        self.start_p = self.config['start_probability']
        self.trans_p = self.config['transition_probability']
        self.emit_p = self.config['emission_probability']
        self.packets = {}
        self.pheap = []

        # build map of all the possible transitions, they are the only ones we need to compute or track
        self.incoming = { st:set() for st in self.state_s }
        for s in self.trans_p:
            for t in self.trans_p[s]:
                if self.trans_p[s][t] > 0. : self.incoming[t].add(s)

    def register_counters(self):
        for label in self.get_dynamic_counter_labels():
            register_dynamic_counter(label, { VITERBI_EVENT })

    def get_dynamic_counter_template_label_mapping(self):
        '''
        Get a dict mapping the dynamic counter labels for this model to the template label
        that is used to specify noise for this model. Dynamic counter labels are
        dependent on the model input.

        We count the following, for all states and packet directions:
          + the total number of emissions
          + the sum of log delays between packet transmission events
          + the sum of squared log delays between packet transmission events (to compute the variance)
          + the total transitions
        '''
        labels = {}

        for state in self.emit_p:
            for direction in self.emit_p[state]:
                template_label = "ExitStreamTrafficModelEmissionCount_<STATE>_<DIRECTION>"
                counter_label = "ExitStreamTrafficModelEmissionCount_{}_{}".format(state, direction)
                labels[counter_label] = template_label

                template_label = "ExitStreamTrafficModelLogDelayTime_<STATE>_<DIRECTION>"
                counter_label = "ExitStreamTrafficModelLogDelayTime_{}_{}".format(state, direction)
                labels[counter_label] = template_label

                template_label = "ExitStreamTrafficModelSquaredLogDelayTime_<STATE>_<DIRECTION>"
                counter_label = "ExitStreamTrafficModelSquaredLogDelayTime_{}_{}".format(state, direction)
                labels[counter_label] = template_label

        for src_state in self.trans_p:
            for dst_state in self.trans_p[src_state]:
                if self.trans_p[src_state][dst_state] > 0.0:
                    template_label = "ExitStreamTrafficModelTransitionCount_<SRCSTATE>_<DSTSTATE>"
                    counter_label = "ExitStreamTrafficModelTransitionCount_{}_{}".format(src_state, dst_state)
                    labels[counter_label] = template_label

        for state in self.start_p:
            if self.start_p[state] > 0.0:
                template_label = "ExitStreamTrafficModelTransitionCount_START_<STATE>"
                counter_label = "ExitStreamTrafficModelTransitionCount_START_{}".format(state)
                labels[counter_label] = template_label

        return labels

    def get_static_counter_template_label_mapping(self):
        '''
        Get a dict mapping the static counter labels for this model to the template label
        that is used to specify noise for this model. Static counter labels are not
        dependent on the model input.
        '''
        static_labels = ['ExitStreamTrafficModelStreamCount',
                         'ExitStreamTrafficModelEmissionCount',
                         'ExitStreamTrafficModelTransitionCount',
                         'ExitStreamTrafficModelLogDelayTime',
                         'ExitStreamTrafficModelSquaredLogDelayTime']
        labels = {}
        for static_label in static_labels:
            labels[static_label] = static_label
        return labels

    def get_all_counter_template_label_mapping(self):
        '''
        Get a dict mapping all static and dynamic counter labels for this model to the
        template label that is used to specify noise for this model.
        '''
        all_labels = self.get_dynamic_counter_template_label_mapping()
        static_labels = self.get_static_counter_template_label_mapping()
        for label in static_labels:
            all_labels[label] = static_labels[label]
        return all_labels

    def get_all_template_labels(self):
        '''
        Return the set of all template labels that are used to specify noise for this model.
        '''
        all_labels = self.get_all_counter_template_label_mapping()
        return set(all_labels.values())

    def get_dynamic_counter_labels(self):
        '''
        Return the set of counters that should be counted for this model,
        but only those whose name depends on the traffic model input.
        '''
        dynamic_labels = self.get_dynamic_counter_template_label_mapping()
        return set(dynamic_labels.keys())

    def get_all_counter_labels(self):
        '''
        Return the set of all counters that will be counted for this model.
        '''
        all_labels = self.get_all_counter_template_label_mapping()
        return set(all_labels.keys())

    def check_noise_config(self, templated_noise_config):
        '''
        Return True if the given templated_noise_config contains the required keys
        for specifying noise for this model, False otherwise.
        '''
        if 'counters' not in templated_noise_config:
            return False

        traffic_noise_valid = True
        for key in templated_noise_config['counters']:
            if key not in self.get_all_template_labels():
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

        label_map = self.get_all_counter_template_label_mapping()
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

        label_map = self.get_all_counter_template_label_mapping()
        for counter_label in label_map:
            template_label = label_map[counter_label]
            # only count the label if it was in the config
            if template_label in templated_noise_config['counters']:
                # traffic counters are all single bin counters
                bins_dict[counter_label] = {'bins': [[0.0, float("inf")]]}

        return bins_dict

    @staticmethod
    def _integer_round(amount, factor):
        '''
        Round amount to a multiple of factor, rounding halfway up.
        '''
        return (((amount + factor/2)/factor)*factor)

    def increment_traffic_counters(self, viterbi_result, secure_counters):
        '''
        Increment the appropriate secure counter labels for this model given the observed
        list of events specifying when bytes were transferred in Tor.
          viterbi_result: the viterbi path through our model given the observed delays
          secure_counters: the SecureCounters object whose counters should get incremented
            as a result of the observed bytes events

        The viterbi_result is encoded as a json string:
          '[["m10s1";"+";35432];["m2s4";"+";0];["m4s2";"-";100];["m4sEnd";"F";0]]'

        This is a list of observations, where each observation has a state name,
        a direction code, and a delay value.

        --old documentation--
        example observations = [('+', 20), ('+', 10), ('+',50), ('+',1000)]
        example viterbi result: Blabbing Blabbing Blabbing Thinking
        then count the following 4:
          - increment 1 for state/observation match:
            Blabbing_+, Blabbing_+, Blabbing_+, Thinking_+
          - increment x where x is log(delay) for each state/observation match:
            Blabbing_+: 2, Blabbing_+: 2, Blabbing_+: 3, Thinking_+: 6
          - increment x*x where x is log(delay) for each state/observation match:
            Blabbing_+: 4, Blabbing_+: 4, Blabbing_+: 9, Thinking_+: 36
          - increment 1 for each state-to-state transition:
            Blabbing_Blabbing, Blabbing_Blabbing, Blabbing_Thinking
        '''

        # python lets you encode with non-default separators, but not decode
        viterbi_result = viterbi_result.replace(';',',')
        path = loads(viterbi_result)

        # the fact that we got an event means that we observed a stream
        secure_counters.increment('ExitStreamTrafficModelStreamCount',
                                  bin=SINGLE_BIN,
                                  inc=1)

        # empty lists are possible, when there was in error in the Tor
        # viterbi code, or when a stream ended with no data sent.
        # if we have an empty list, the following loop will not execute.
        for i, packet in enumerate(path):
            if len(packet) < 3:
                continue

            state, direction, delay = str(packet[0]), str(packet[1]), int(packet[2])

            # delay of 0 indicates the packets were observed at the same time
            # log(x=0) is undefined, and log(x<1) is negative
            # we don't want to count negatives, so override delay if needed
            ldelay = 0 if delay < 1 else int(round(math.log(delay)))

            secure_counters.increment('ExitStreamTrafficModelEmissionCount',
                                      bin=SINGLE_BIN,
                                      inc=1)
            label = 'ExitStreamTrafficModelEmissionCount_{}_{}'.format(state, direction)
            secure_counters.increment(label,
                                      bin=SINGLE_BIN,
                                      inc=1)

            secure_counters.increment('ExitStreamTrafficModelLogDelayTime',
                                      bin=SINGLE_BIN,
                                      inc=ldelay)
            label = 'ExitStreamTrafficModelLogDelayTime_{}_{}'.format(state, direction)
            secure_counters.increment(label,
                                      bin=SINGLE_BIN,
                                      inc=ldelay)

            secure_counters.increment('ExitStreamTrafficModelSquaredLogDelayTime',
                                      bin=SINGLE_BIN,
                                      inc=ldelay*ldelay)
            label = 'ExitStreamTrafficModelSquaredLogDelayTime_{}_{}'.format(state, direction)
            secure_counters.increment(label,
                                      bin=SINGLE_BIN,
                                      inc=ldelay*ldelay)

            if i == 0: # track starting transitions
                label = 'ExitStreamTrafficModelTransitionCount_START_{}'.format(state)
                secure_counters.increment(label,
                                          bin=SINGLE_BIN,
                                          inc=1)

            # track transitions for all but the final state
            if (i + 1) < len(path) and len(path[i + 1]) >= 3:
                next_state = str(path[i + 1][0])
                secure_counters.increment('ExitStreamTrafficModelTransitionCount',
                                          bin=SINGLE_BIN,
                                          inc=1)
                label = 'ExitStreamTrafficModelTransitionCount_{}_{}'.format(state, next_state)
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
            label = "ExitStreamTrafficModelTransitionCount_START_{}".format(state)
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
                label = "ExitStreamTrafficModelTransitionCount_{}_{}".format(src_state, dst_state)
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

        e_count, e_mu, e_sigma = {}, {}, {}
        for state in self.emit_p:
            # first we go through and convert tallies to valid non-negative values
            e_count[state], e_mu[state], e_sigma[state] = {}, {}, {}
            for direction in self.emit_p[state]:
                sd_label = "ExitStreamTrafficModelEmissionCount_{}_{}".format(state, direction)
                if sd_label in tallies and tallies[sd_label] > 0.0:
                    e_count[state][direction] = float(tallies[sd_label])
                else:
                    e_count[state][direction] = 0.0

                mu_label = "ExitStreamTrafficModelLogDelayTime_{}_{}".format(state, direction)
                if mu_label in tallies and tallies[mu_label] > 0.0 and e_count[state][direction] > 0.0:
                    e_mu[state][direction] = float(tallies[mu_label])/float(e_count[state][direction])
                else:
                    e_mu[state][direction] = 0.0

                ss_label = "ExitStreamTrafficModelSquaredLogDelayTime_{}_{}".format(state, direction)
                if ss_label in tallies and tallies[ss_label] > 0.0 and sd_label in tallies and tallies[sd_label] > 0.0:
                    obs_var = float(tallies[ss_label])/float(tallies[sd_label])
                    obs_var -= e_mu[state][direction]**2
                else:
                    obs_var = 0.0

                # rounding errors or noise can make a small positive variance look negative
                # setting a small "sane default" for this case
                if obs_var < math.sqrt(0.01):
                    e_sigma[state][direction] = 0.01
                else: # No rounding errors, do the math
                    e_sigma[state][direction] = math.sqrt(obs_var)

            # normalize counts to ensure the probabilities sum to 100%,
            # or set to 0 if we had absolutely no positive counts.
            e_total = float(sum(e_count[state].values()))
            for direction in e_count[state]:
                if e_total > 0.0:
                    e_count[state][direction] = e_count[state][direction]/e_total
                else:
                    e_count[state][direction] = 0.0

        return e_count, e_mu, e_sigma

    def __update_emission_probabilities(self, e_count, e_mu, e_sigma, inertia):
        '''
        Update the emission probabilities based on the normalized probabilities from the
        tally count results.
        Updates self.emit_p in place and also returns a reference to it.
        '''

        for state in self.emit_p:
            # apply inertia
            for direction in self.emit_p[state]:
                (dp, mu, sigma) = self.emit_p[state][direction]

                dp_new = inertia * dp + (1.0-inertia) * e_count[state][direction]
                mu_new = inertia * mu + (1.0-inertia) * e_mu[state][direction]
                sigma_new = inertia * sigma + (1.0-inertia) * e_sigma[state][direction]

                self.emit_p[state][direction] = (dp_new, mu_new, sigma_new)

            # be extra sure our new probs sum to 1.0
            e_total = float(sum([v[0] for v in self.emit_p[state].values()]))
            if e_total > 0.0:
                for direction in self.emit_p[state]:
                    (dp, mu, sigma) = self.emit_p[state][direction]
                    dp_new = dp/e_total
                    self.emit_p[state][direction] = (dp_new, mu, sigma)
            else:
                logging.warning("BUG: we have no positive emission probabilities in updated model for state {}."
                    .format(state))

        return self.emit_p

    def update_from_tallies(self, tallies, trans_inertia=0.5, emit_inertia=0.5):
        '''
        Given the (noisy) aggregated tallied counter values for this model, compute the updated model:
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
        e_count, e_mu, e_sigma = self.__normalize_emission_tallies(tallies)
        self.emit_p = self.__update_emission_probabilities(e_count, e_mu, e_sigma, emit_inertia)

        updated_model_config = {
            'state_space': self.state_s,
            'observation_space': self.obs_s,
            'start_probability': self.start_p,
            'transition_probability': self.trans_p,
            'emission_probability': self.emit_p
        }

        return updated_model_config
