'''
Created on Dec 11, 2016

@author: rob
'''
from math import log

from privcount.counter import register_dynamic_counter, BYTES_EVENT

def check_traffic_model_config(model_config):
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
        self.start_p = self.config['start_p']
        self.trans_p = self.config['trans_p']
        self.emit_p = self.config['emit_p']

    def register_counters(self):
        for label in self.get_dynamic_counter_labels():
            register_dynamic_counter(label, { BYTES_EVENT })

    def get_dynamic_counter_labels(self):
        '''
        Return the set of counters that should be counted for this model,
        but only those whose name depends on the traffic model.
        Use get_counter_labels() to get the set of all counters used to
        count this traffic model.
        '''
        labels = []
        for state in self.emit_p:
            for direction in self.emit_p[state]:
                labels.append("TrafficModelTotalEmissions_{}{}".format(state, direction))
                labels.append("TrafficModelTotalDelay_{}{}".format(state, direction))
        for src_state in self.trans_p:
            for dst_state in self.trans_p[src_state]:
                labels.append("TrafficModelTotalTransitions_{}_{}".format(src_state, dst_state))
        return labels

    def get_all_counter_labels(self):
        '''
        Return the set of counters that should be counted for this model.
        We should count the following, for all states and packet directions:
          + the total number of emissions
          + the total delay between packet transmission events
          + the total transitions
        '''
        dynamic_labels = self.get_dynamic_counter_labels()
        static_labels = ["TrafficModelTotalEmissions", "TrafficModelTotalDelay", "TrafficModelTotalTransitions"]
        return static_labels + dynamic_labels


    def run_viterbi(self, obs):
        '''
        Given a list of packet observations of the form ('+' or '-', delay_time), e.g.:
            [('+', 10), ('+', 20), ('+', 50), ('+', 1000)]
        Run the viterbi dynamic programming algorithm to determine which path through the HMM has the highest probability, i.e.,, closest match to these observations.
        '''
        V = [{}]
        for st in self.states:
            # updated emit_p here
            (direction, delay) = obs[0]
            (dp, dlam) = self.emit_p[st][direction]
            fitprob = log(dp) + log(dlam) - (delay*dlam)
            # replaced the following line
            #V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
            V[0][st] = {"prob": log(self.start_p[st]) + fitprob, "prev": None}
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in self.states:
                max_tr_prob = max(V[t-1][prev_st]["prob"]+log(self.trans_p[prev_st][st]) for prev_st in self.states)
                for prev_st in self.states:
                    if V[t-1][prev_st]["prob"] + log(self.trans_p[prev_st][st]) == max_tr_prob:
                        # updated emit_p here
                        (direction, delay) = obs[t]
                        (dp, dlam) = self.emit_p[st][direction]
                        fitprob = log(dp) + log(dlam) - (delay*dlam)
                        # replaced the following line
                        #max_prob = max_tr_prob * emit_p[st][obs[t]]
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

        #print 'The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob
        return opt # list of highest probable states, in order
