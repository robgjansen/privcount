'''
Created on Dec 11, 2016

@author: rob

See LICENSE for licensing information
'''
import math
import logging

from time import time, clock
from heapq import heappush, heappop

from privcount.counter import register_dynamic_counter, VITERBI_EVENT, SecureCounters
SINGLE_BIN = SecureCounters.SINGLE_BIN

def check_traffic_model_config(model_config):
    '''
    Return True if the given model_config contains the required keys for incrementing counters
    for this model, False otherwise.
    '''
    traffic_model_valid = True
    for k in ['states', 'emission_probability', 'transition_probability', 'start_probability']:
        if k not in model_config:
            traffic_model_valid = False
    return traffic_model_valid

class TrafficModel(object):
    '''
    A class that represents a hidden markov model (HMM).
    See `test/traffic.model.json` for a simple traffic model that this class can represent.
    '''

    # the approximate number of payload bytes for a packet
    PACKET_BYTE_COUNT = 1434
    # assume a packet arrived at the same time if it arrived
    # within this many microseconds
    PACKET_ARRIVAL_TIME_TOLERENCE = long(100)

    # the maximum number of packets we will handle in a stream before issuing
    # a delay warning. On my relays, this takes 10 seconds of processing time
    # on a large model
    MAX_STREAM_PACKET_COUNT = 10000
    # the maximum number of seconds we will take to process a stream before
    # issuing a delay warning
    MAX_STREAM_PROCESSING_TIME = 10.0
    # the maximum number of seconds we are willing to store cells for a stream
    # before considering it stale and evicting its stored cell data
    STREAM_EVICT_TIME = 3600.0

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
        self.states = self.config['states']
        self.start_p = self.config['start_probability']
        self.trans_p = self.config['transition_probability']
        self.emit_p = self.config['emission_probability']
        self.packets = {}
        self.pheap = []

        # build map of all the possible transitions, they are the only ones we need to compute or track
        self.incoming = { st:set() for st in self.states }
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
        static_labels = ['ExitStreamTrafficModelEmissionCount',
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
        for key in self.get_all_template_labels():
            if key not in templated_noise_config['counters']:
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
            noise_dict[counter_label] = templated_noise_config['counters'][template_label]

        return noise_dict

    def get_bins_init_config(self):
        '''
        Returns the initial bin values for all counters counted by this model.
        '''
        bins_dict = {}
        for counter_label in self.get_all_counter_labels():
            # we count all single values, so bins should all represent a single count
            bins_dict[counter_label] = {'bins': [[0.0, float("inf")]]}

        return bins_dict

    def _run_viterbi(self, bundles):
        '''
        Each bundle in a list of packet 'bundles' represents many packets that arrived at the same time:
            packet_bundle = [is_sent, micros_since_prev_cell, bundle_ts, num_packets, payload_bytes_last_packet]
        Run the viterbi dynamic programming on a list of such bundles.

        Viterbi normally expects a list of packet observations of the form ('+' or '-', delay_time), e.g.:
            [('+', 10), ('+', 20), ('+', 50), ('+', 1000)]
        Viterbi determines which path through the HMM has the highest probability, i.e., closest match to these observations.
        '''
        SQRT_2_PI = math.sqrt(2*math.pi)
        V = [{}]
        total_num_obs = 0
        bundle_i = 0

        current_bundle = bundles[bundle_i]
        current_bundle_num_packets = current_bundle[3]

        # figure out probabilities for starting state
        # first get info from first observation
        if current_bundle[0] == None:
            direction = 'F'
        else:
            direction = '-' if current_bundle[0] else '+'

        delay = current_bundle[1]
        if delay <= 2: dx = 1
        else: dx = int(math.exp(int(math.log(delay))))

        # we 'used up' the first observation
        total_num_obs += 1
        current_bundle_num_packets -= 1

        for st in self.states:
            if st in self.start_p and self.start_p[st] > 0 and \
                    st in self.emit_p and direction in self.emit_p[st]:
                # updated emit_p here
                (dp, mu, sigma) = self.emit_p[st][direction]
                delay_logp = -math.log( dx * sigma * SQRT_2_PI ) - 0.5 * ( ( math.log( dx ) - mu ) / sigma ) ** 2
                fitprob = math.log(dp) + delay_logp
                V[0][st] = {"prob": math.log(self.start_p[st]) + fitprob, "prev": None}
            else:
                V[0][st] = {"prob": float("-inf"), "prev": None }

        # Run Viterbi when t > 0
        t = 0 # starts at 0 but is immediately incremented below if we have another packet
        while True:
            # get the next packet, which could be in the same 'bundled' set of packets
            while current_bundle is not None and current_bundle_num_packets <= 0:
                # current bundle was used up
                current_bundle = None
                bundle_i += 1
                # ran out of packets, get the next bundle
                if bundle_i < len(bundles):
                    current_bundle = bundles[bundle_i]
                    if current_bundle:
                        current_bundle_num_packets = current_bundle[3]

            # break if we have no more packets
            if current_bundle is None:
                break

            # this represents the next observation
            t += 1
            if current_bundle[0] == None:
                direction = 'F'
            else:
                direction = '-' if current_bundle[0] else '+'
            delay = current_bundle[1]
            if delay <= 2: dx = 1
            else: dx = int(math.exp(int(math.log(delay))))

            # we 'used up' this observation
            current_bundle_num_packets -= 1
            total_num_obs += 1

            V.append({})
            for st in self.states:
                max_tr_prob = max(V[t-1][prev_st]["prob"]+math.log(self.trans_p[prev_st][st]) for prev_st in self.incoming[st])
                for prev_st in self.incoming[st]:
                    if V[t-1][prev_st]["prob"] + math.log(self.trans_p[prev_st][st]) == max_tr_prob:
                        if direction not in self.emit_p[st]:
                            V[t][st] = {"prob": float("-inf"), "prev": prev_st}
                            break
                        (dp, mu, sigma) = self.emit_p[st][direction]
                        delay_logp = -math.log( dx * sigma * SQRT_2_PI ) - 0.5 * ( ( math.log( dx ) - mu ) / sigma ) ** 2
                        fitprob = math.log(dp) + delay_logp
                        max_prob = max_tr_prob + fitprob
                        V[t][st] = {"prob": max_prob, "prev": prev_st}
                        break

        #for line in dptable(V):
        #    print line
        opt = []
        # The highest probability
        max_prob = max(value["prob"] for value in V[-1].values())
        previous = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] == max_prob:
                opt.append(st)
                previous = st
                break
        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        if len(opt) != total_num_obs:
            logging.warning("We saw stream with {} observations but computed {} states".format(total_num_obs, len(opt)))

        #print 'The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob
        return opt # list of highest probable states, in order

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
          likliest_states: the likliest path through our model given the observed delays
          secure_counters: the SecureCounters object whose counters should get incremented
            as a result of the observed bytes events

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
        # TODO FIXME XXX this should be the only remaining function to edit
        return

        num_states = len(likliest_states)
        i = 0

        for current_bundle in bundles:
            dir_code = '-' if current_bundle[0] else '+'
            # delay is in microseconds
            delay = current_bundle[1]
            num_packets_in_bundle = current_bundle[3]

            while num_packets_in_bundle > 0:
                state = likliest_states[i]

                # delay of 0 indicates the packets were observed at the same time
                # log(x=0) is undefined, and log(x<1) is negative
                # we don't want to count negatives, so override delay if needed
                ldelay = 0 if delay < 1 else int(math.log(delay))

                secure_counters.increment('ExitStreamTrafficModelEmissionCount',
                                          bin=SINGLE_BIN,
                                          inc=1)
                label = 'ExitStreamTrafficModelEmissionCount_{}_{}'.format(state, dir_code)
                secure_counters.increment(label,
                                          bin=SINGLE_BIN,
                                          inc=1)

                secure_counters.increment('ExitStreamTrafficModelLogDelayTime',
                                          bin=SINGLE_BIN,
                                          inc=ldelay)
                label = 'ExitStreamTrafficModelLogDelayTime_{}_{}'.format(state, dir_code)
                secure_counters.increment(label,
                                          bin=SINGLE_BIN,
                                          inc=ldelay)

                secure_counters.increment('ExitStreamTrafficModelSquaredLogDelayTime',
                                          bin=SINGLE_BIN,
                                          inc=ldelay*ldelay)
                label = 'ExitStreamTrafficModelSquaredLogDelayTime_{}_{}'.format(state, dir_code)
                secure_counters.increment(label,
                                          bin=SINGLE_BIN,
                                          inc=ldelay*ldelay)

                if i == 0: # track starting transitions
                    label = 'ExitStreamTrafficModelTransitionCount_START_{}'.format(state)
                    secure_counters.increment(label,
                                              bin=SINGLE_BIN,
                                              inc=1)
                if (i+1) < num_states:
                    next_state = likliest_states[i+1]
                    secure_counters.increment('ExitStreamTrafficModelTransitionCount',
                                              bin=SINGLE_BIN,
                                              inc=1)
                    label = 'ExitStreamTrafficModelTransitionCount_{}_{}'.format(state, next_state)
                    secure_counters.increment(label,
                                              bin=SINGLE_BIN,
                                              inc=1)

                # upkeep
                num_packets_in_bundle -= 1
                i += 1

    def update_from_tallies(self, tallies, trans_inertia=0.1, emit_inertia=0.1):
        '''
        Given the (noisy) aggregated tallied counter values for this model, compute the updated model:
        + Transition probabilities - trans_p[s][t] = inertia*trans_p[s][t] + (1-inertia)*(tally result)
        + Emission probabilities - emit_p[s][d] <- dp', mu', sigma' where
          dp' <- emit_inertia * dp + (1-emit_inertia)*(state_direction count / state count)
          mu' <- emit_inertia * mu + (1-emit_inertia)*(log-delay sum / state_direction count)
          sigma' <- emit_inertia * sigma + (1-emit_inertia)*sqrt(avg. of squares - square of average)
        '''
        count, trans_count, obs_trans_p = {}, {}, {}
        for src_state in self.trans_p:
            count[src_state] = 0
            trans_count[src_state] = {}
            for dst_state in self.trans_p[src_state]:
                trans_count[src_state][dst_state] = 0
                src_dst_label = "ExitStreamTrafficModelTransitionCount_{}_{}".format(src_state, dst_state)
                if src_dst_label in tallies:
                    val = tallies[src_dst_label]
                    trans_count[src_state][dst_state] = val
                    count[src_state] += val

            obs_trans_p[src_state] = {}
            for dst_state in self.trans_p[src_state]:
                if count[src_state] > 0:
                    obs_trans_p[src_state][dst_state] = float(trans_count[src_state][dst_state])/float(count[src_state])
                else:
                    obs_trans_p[src_state][dst_state] = 0.0

        obs_dir_emit_count, obs_mu, obs_sigma = {}, {}, {}
        for state in self.emit_p:
            obs_dir_emit_count[state], obs_mu[state], obs_sigma[state] = {}, {}, {}
            for direction in self.emit_p[state]:
                sd_label = "ExitStreamTrafficModelEmissionCount_{}_{}".format(state, direction)
                if sd_label in tallies:
                    obs_dir_emit_count[state][direction] = tallies[sd_label]
                else:
                    obs_dir_emit_count[state][direction] = 0

                mu_label = "ExitStreamTrafficModelLogDelayTime_{}_{}".format(state, direction)
                if mu_label in tallies and obs_dir_emit_count[state][direction] > 0:
                    obs_mu[state][direction] = float(tallies[mu_label])/float(obs_dir_emit_count[state][direction])
                else:
                    obs_mu[state][direction] = 0.0

                ss_label = "ExitStreamTrafficModelSquaredLogDelayTime_{}_{}".format(state, direction)
                if ss_label in tallies and sd_label in tallies and tallies[sd_label] > 0:
                    obs_var = float(tallies[ss_label])/float(tallies[sd_label])
                    obs_var -= obs_mu[state][direction]**2
                else:
                    obs_var = 0.0

                # rounding errors or noise can make a small positive variance look negative
                # setting a small "sane default" for this case
                if obs_var < math.sqrt(0.01):
                    obs_sigma[state][direction] = 0.01
                else: # No rounding errors, do the math
                    obs_sigma[state][direction] = math.sqrt(obs_var)

        for src_state in self.trans_p:
            for dst_state in self.trans_p[src_state]:
                self.trans_p[src_state][dst_state] = trans_inertia * self.trans_p[src_state][dst_state] + (1-trans_inertia) * obs_trans_p[src_state][dst_state]

            state = src_state
            for direction in self.emit_p[state]:
                (dp, mu, sigma) = self.emit_p[state][direction]

                if state in count and count[state] > 0:
                    dp_new = emit_inertia * dp + (1-emit_inertia)*float(obs_dir_emit_count[state][direction])/float(count[state])
                else:
                    dp_new = emit_inertia * dp
                mu_new = emit_inertia * mu + (1-emit_inertia)*obs_mu[state][direction]
                sigma_new = emit_inertia * sigma + (1-emit_inertia)*obs_sigma[state][direction]

                self.emit_p[state][direction] = (dp_new, mu_new, sigma_new)

        # handle start probabilities.
        s_label, s_count = {}, {}
        start_total = 0
        for state in self.start_p:
            if self.start_p[state] > 0.0:
                s_label = "ExitStreamTrafficModelTransitionCount_START_{}".format(state)
                if s_label in tallies:
                    s_count[state] = tallies[s_label]
                else:
                    s_count[state] = 0
                start_total += s_count[state]
        for state in self.start_p:
            if start_total > 0.0:
                self.start_p[state] = trans_inertia * self.start_p[state] + (1-trans_inertia) * float(s_count[state])/float(start_total)
            else:
                self.start_p[state] = trans_inertia * self.start_p[state]

        updated_model_config = {
            'states': self.states,
            'start_probability': self.start_p,
            'transition_probability': self.trans_p,
            'emission_probability': self.emit_p
        }
        return updated_model_config
