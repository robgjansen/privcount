'''
Created on Dec 11, 2016

@author: rob
'''
import math

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
        self.start_p = self.config['start_probability']
        self.trans_p = self.config['transition_probability']
        self.emit_p = self.config['emission_probability']

        # build map of all the possible transitions, they are the only ones we need to compute or track
        self.incoming = { st:set() for st in self.states }
        for s in self.trans_p:
            for t in self.trans_p[s]:
                if self.trans_p[s][t] > 0. : self.incoming[t].add(s)

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
                labels.append("TrafficModelTotalLogDelay_{}{}".format(state, direction))
                labels.append("TrafficModelTotalSquaredLogDelay_{}{}".format(state, direction))
        for src_state in self.trans_p:
            for dst_state in self.trans_p[src_state]:
                if self.trans_p[src_state][dst_state] > 0.0:
                    labels.append("TrafficModelTotalTransitions_{}_{}".format(src_state, dst_state))
        for state in self.start_p:
            if self.start_p[state] > 0.0:
                labels.append("TrafficModelTotalTransitions_START_{}".format(state))

        return labels

    def get_all_counter_labels(self):
        '''
        Return the set of counters that should be counted for this model.
        We should count the following, states and packet directions:
          + the total number of emissions
          + the sum of log delays between packet transmission events
          + the sum of squared log delays between packet transmission events (to compute the variance)
          + the total transitions
        '''
        static_labels = ["TrafficModelTotalEmissions", "TrafficModelTotalTransitions",
                         "TrafficModelTotalLogDelay", "TrafficModelTotalSquaredLogDelay"]
        dynamic_labels = self.get_dynamic_counter_labels()
        return static_labels + dynamic_labels

    def run_viterbi(self, obs):
        '''
        Given a list of packet observations of the form ('+' or '-', delay_time), e.g.:
            [('+', 10), ('+', 20), ('+', 50), ('+', 1000)]
        Run the viterbi dynamic programming algorithm to determine which path through the HMM has the highest probability, i.e.,, closest match to these observations.
        '''
        SQRT_2_PI = math.sqrt(2*math.pi)
        V = [{}]
        for st in self.states:
            if st in self.start_p and self.start_p[st] > 0:
                # updated emit_p here
                (direction, delay) = obs[0]
                (dp, mu, sigma) = self.emit_p[st][direction]
                if delay <= 2: dx = 1
                else: dx = int(math.exp(int(math.log(delay))))
                delay_logp = -math.log( dx * sigma * SQRT_2_PI ) - 0.5 * ( ( math.log( dx ) - mu ) / sigma ) ** 2
                fitprob = math.log(dp) + delay_logp
                # replaced the following line
                #V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
                V[0][st] = {"prob": math.log(self.start_p[st]) + fitprob, "prev": None}
            else:
                V[0][st] = {"prob": float("-inf"), "prev": None }
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in self.states:
                max_tr_prob = max(V[t-1][prev_st]["prob"]+math.log(self.trans_p[prev_st][st]) for prev_st in self.incoming[st])
                for prev_st in self.incoming[st]:
                    if V[t-1][prev_st]["prob"] + math.log(self.trans_p[prev_st][st]) == max_tr_prob:
                        # updated emit_p here
                        (direction, delay) = obs[t]
                        if direction not in self.emit_p[st]:
                            V[t][st] = {"prob": float("-inf"), "prev": prev_st}
                            break
                        (dp, mu, sigma) = self.emit_p[st][direction]
                        if delay <= 2: dx = 1
                        else: dx = int(math.exp(int(math.log(delay))))
                        delay_logp = -math.log( dx * sigma * SQRT_2_PI ) - 0.5 * ( ( math.log( dx ) - mu ) / sigma ) ** 2
                        fitprob = math.log(dp) + delay_logp
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

    def update_from_counters(counters, trans_inertia=0.1, emit_inertia=0.1):
        '''
        Given the (noisy) aggregated counter values for this model, compute the updated model:
        + Transition probabilities - trans_p[s][t] = inertia*trans_p[s][t] + (1-inertia)*(tally result)
        + Emission probabilities - emit_p[s][d] <- dp', mu', sigma' where
          dp' <- emit_inertia * dp + (1-emit_inertia)*(state_direction count / state count)
          mu' <- emit_inertia * mu + (1-emit_inertia)*(log-delay sum / state_direction count)
          sigma' <- emit_inertia * sigma + (1-emit_inertia)*sqrt(avg. of squares - square of average)
        '''
        obs_trans_p = { }
        count = { }
        for s in self.states:
            count[s] = 0
            obs_trans_p[s] = { }
            for t in self.trans_p[s]:
                st_label = "TrafficModelTotalTransitions_{}_{}".format(s,t)
                count[s] += counters[st_label]
                obs_trans_p[s][t] = counters[st_label]
            for t in self.trans_p[s]:
                obs_trans_p[s][t] = float(obs_trans_p[s][t])/count[s]

        obs_dir_emit_count = { }
        obs_mu = { }
        obs_sigma = { }
        for s in self.states:
            obs_dir_emit_count[s] = { }
            obs_mu[s] = { }
            for d in self.emit_p[s]:
                sd_label = "TrafficModelTotalEmissions_{}{}".format(s,d)
                obs_dir_emit_count[s][d] = counters[sd_label]
                mu_label = "TrafficModelTotalLogDelay_{}{}".format(s,d)
                obs_mu[s][d] = float(counters[mu_label])/obs_dir_emit_count[s][d]
                ss_label = "TrafficModelTotalSquaredLogDelay_{}{}".format(s,d)
                obs_var = float(counters[ss_label])/counters[sd_label] - obs_mu[s][d]**2
                # rounding errors or noise can make a small positive variance look negative
                # setting a small "sane default" for this case
                if obs_var < math.sqrt(0.01):
                    obs_sigma[s][d] = 0.01
                else: # No rounding errors, do the math
                    obs_sigma[s][d] = math.sqrt(obs_var)

        for s in self.states:
            for t in self.trans_p[s]:
                self.trans_p[s][t] = trans_inertia * self.trans_p[s][t] + (1-trans_inertia) * obs_trans_p[s][t]

            for d in self.emit_p[s]:
                (dp, mu, sigma) = self.emit_p[s][d]
                self.emit_p[s][d] = (emit_inertia * dp + (1-emit_inertia)*float(obs_dir_emit_count[s][d])/count[s],
                                     emit_inertia * mu + (1-emit_inertia)*obs_mu[s][d],
                                     emit_inertia * sigma + (1-emit_inertia)*obs_sigma[s][d])
        # handle start probabilities.
        s_label = { }
        s_count = { }
        start_total = 0
        for s in self.start_p:
            s_label = "TrafficModelTotalTransitions_START_{}".format(s)
            s_count[s] = counters[s_label]
            start_total += s_count[s]
        for s in self.start_p:
            start_p[s] = trans_inertia * self.start_p[s] + (1-trans_inertia) * (float(s_count[s])/start_total)

        return (self.states, self.start_p, self.trans_p, self.emit_p)
