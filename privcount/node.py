'''
Created on Dec 6, 2016

@author: teor
'''

import logging

from time import time

from privcount.config import normalise_path
from privcount.counter import check_counters_config, check_noise_weight_config, combine_counters, CollectionDelay, float_accuracy, add_counter_limits_to_config
from privcount.log import format_delay_time_until, format_elapsed_time_since
from privcount.statistics_noise import DEFAULT_SIGMA_TOLERANCE
from privcount.traffic_model import TrafficModel, check_traffic_model_config

def get_remaining_rounds(num_phases, continue_config):
        '''
        If the TS is configured to continue collecting a limited number of
        rounds, return the number of rounds. Otherwise, if it will continue
        forever, return None.
        '''
        if num_phases == 0:
            return 1
        if isinstance(continue_config, bool):
            if continue_config:
                return None
            else:
                return 0
        else:
            return continue_config - num_phases

def continue_collecting(num_phases, continue_config):
        '''
        If the TS is configured to continue collecting more rounds,
        return True. Otherwise, return False.
        '''
        if num_phases == 0:
            return True
        if isinstance(continue_config, bool):
            return continue_config
        else:
            return continue_config > num_phases

def log_tally_server_status(status):
    '''
    clients must only use the expected end time for logging: the tally
    server may end the round early, or extend it slightly to allow for
    network round trip times
    '''
    # until the collection round starts, the tally server doesn't know when it
    # is expected to end
    expected_end_msg = ""
    if 'expected_end_time' in status:
        stopping_ts = status['expected_end_time']
        # we're waiting for the collection to stop
        if stopping_ts > time():
            expected_end_msg = ", expect collection to end in {}".format(format_delay_time_until(stopping_ts, 'at'))
        # we expect the collection to have stopped, and the TS should be
        # collecting results
        else:
            expected_end_msg = ", expect collection has ended for {}".format(format_elapsed_time_since(stopping_ts, 'since'))
    logging.info("--server status: PrivCount is {} for {}{}".format(status['state'], format_elapsed_time_since(status['time'], 'since'), expected_end_msg))
    t, r = status['dcs_total'], status['dcs_required']
    a, i = status['dcs_active'], status['dcs_idle']
    logging.info("--server status: DataCollectors: have {}, need {}, {}/{} active, {}/{} idle".format(t, r, a, t, i, t))
    t, r = status['sks_total'], status['sks_required']
    a, i = status['sks_active'], status['sks_idle']
    logging.info("--server status: ShareKeepers: have {}, need {}, {}/{} active, {}/{} idle".format(t, r, a, t, i, t))
    if continue_collecting(status['completed_phases'],
                           status['continue']):
        rem = get_remaining_rounds(status['completed_phases'],
                                   status['continue'])
        if rem is not None:
            continue_str = "continue for {} more rounds".format(rem)
        else:
            continue_str = "continue indefinitely"
        next_start_time = status['delay_until']
        if next_start_time > time():
            next_round_str = " in {}".format(format_delay_time_until(
                                                 next_start_time, 'at'))
        else:
            next_round_str = " as soon as clients are ready"
    else:
        continue_str = "stop"
        next_round_str = " after this collection round"
    logging.info("--server status: Rounds: completed {}, configured to {} collecting{}"
                 .format(status['completed_phases'],
                         continue_str,
                         next_round_str))

class PrivCountNode(object):
    '''
    A mixin class that hosts common functionality for PrivCount client and
    server factories: TallyServer, ShareKeeper, and DataCollector.
    '''

    def __init__(self, config_filepath):
        '''
        Initialise the common data structures used by all PrivCount nodes.
        '''
        self.config_filepath = normalise_path(config_filepath)
        self.config = None
        self.collection_delay = CollectionDelay()

    def load_state(self):
        '''
        Load the state from the saved state file
        Return the loaded state, or None if there is no state file
        '''
        # load any state we may have from a previous run
        state_filepath = normalise_path(self.config['state'])
        if os.path.exists(state_filepath):
            with open(state_filepath, 'r') as fin:
                state = pickle.load(fin)
                return state
        return None

    def dump_state(self, state):
        '''
        Dump the state dictionary to a saved state file.
        If state is none or an empty dictionary, do not write a file.
        '''
        if state is None or len(state.keys()) == 0:
            return
        state_filepath = normalise_path(self.config['state'])
        with open(state_filepath, 'w') as fout:
            pickle.dump(state, fout)

    def get_secret_handshake_path(self):
        '''
        Return the path of the secret handshake key file, or None if the config
        has not been loaded.
        Called by the protocol after a connection is opened.
        '''
        # The config must have been loaded by this point:
        # - the server reads the config before opening a listener port
        # - the clients read the config before opening a connection
        assert self.config
        # The secret handshake path should be loaded (or assigned a default)
        # whenever the config is loaded
        return self.config['secret_handshake']

    @staticmethod
    def get_valid_sigma_decrease_tolerance(conf):
        '''
        Read sigma_decrease_tolerance from conf (if present), and check that
        it is within a valid range.
        Returns the configured sigma tolerance, or the default tolerance.
        Asserts on failure.
        '''
        tolerance = conf.get('sigma_decrease_tolerance',
                             DEFAULT_SIGMA_TOLERANCE)

        # we can't guarantee that floats are transmitted with any more
        # accuracy than approximately 1 part in 1e-14, due to python
        # float to string conversion
        # so we limit the tolerance to an absolute value of ~1e-14,
        # which assumes the sigma values are close to 1.
        # larger sigma values should have a larger absolute limit, because
        # float_accuracy() is a proportion of the value,
        # but we can't do that calculation here
        assert tolerance >= float_accuracy()
        return tolerance

    # 24 hours is a safe default for protecting user activity between rounds
    DEFAULT_DELAY_PERIOD = 24*60*60

    @staticmethod
    def get_valid_delay_period(conf):
        '''
        Validate and return the delay period from conf
        Returns a (potentially modified) valid value.
        '''
        # Validity checks
        if 'delay_period' not in conf:
            logging.warning("delay_period not specified, using default %d",
                            PrivCountNode.DEFAULT_DELAY_PERIOD)
            return PrivCountNode.DEFAULT_DELAY_PERIOD
        if conf['delay_period'] < 0:
            logging.warning("delay_period %d invalid, using default %d",
                            conf['delay_period'],
                            PrivCountNode.DEFAULT_DELAY_PERIOD)
            return PrivCountNode.DEFAULT_DELAY_PERIOD
        # Privacy warning (but keep the value)
        if conf['delay_period'] < PrivCountNode.DEFAULT_DELAY_PERIOD:
            logging.warning("delay_period %d is less than the default %d, this only protects %d seconds of user activity",
                            conf['delay_period'],
                            PrivCountNode.DEFAULT_DELAY_PERIOD,
                            conf['delay_period'])
        # If it passes all the checks
        return conf['delay_period']

class PrivCountServer(PrivCountNode):
    '''
    A mixin class that hosts common functionality for PrivCount server
    factories: TallyServer.
    (Since there is only one server factory class, this class only hosts
    generic functionality that is substantially similar to PrivCountClient,
    but not identical - if it were identical, it would go in PrivCountNode.)
    '''

    def __init__(self, config_filepath):
        '''
        Initialise the common data structures used by all PrivCount clients.
        '''
        PrivCountNode.__init__(self, config_filepath)

    @staticmethod
    def get_valid_sigma_decrease_tolerance(conf):
        '''
        Read sigma_decrease_tolerance from conf (if present), and check that
        it is withing a valid range, taking the noise allocation config into
        account (if present).
        '''
        tolerance = PrivCountNode.get_valid_sigma_decrease_tolerance(conf)

        # it makes no sense to have a sigma decrease tolerance that is
        # less than the sigma calculation tolerance
        # (if we use hard-coded sigmas, calculation accuracy is not
        # an issue - skip this check)
        if 'sigma_tolerance' in conf['noise'].get('privacy',{}):
            assert (tolerance >=
                    conf['noise']['privacy']['sigma_tolerance'])
        elif 'privacy' in conf['noise']:
            assert (tolerance >=
                    DEFAULT_SIGMA_TOLERANCE)
        else:
            # no extra checks
            pass

        return tolerance

class PrivCountClient(PrivCountNode):
    '''
    A mixin class that hosts common functionality for PrivCount client
    factories: ShareKeeper and DataCollector.
    '''

    def __init__(self, config_filepath):
        '''
        Initialise the common data structures used by all PrivCount clients.
        '''
        PrivCountNode.__init__(self, config_filepath)
        self.start_config = None
        # the collect period supplied by the tally server
        self.collect_period = None
        # the noise config used to start the most recent round
        self.last_noise_config = None
        # the start time of the most recent round
        self.collection_start_time = None

    def set_server_status(self, status):
        '''
        Called by protocol
        status is a dictionary containing server status information
        '''
        log_tally_server_status(status)

    def set_round_start(self, start_config):
        '''
        Set the round start variables:
         - the delay period after this round,
         - the noise config,
         - the start time,
         based on the start config and loaded config.
        '''
        self.collect_period = start_config['collect_period']
        self.last_noise_config = start_config['noise']
        self.collection_start_time = time()

    def check_start_config(self, start_config):
        '''
        Perform the common client checks on the start config.
        Return the combined counters if the start_config is valid,
        or None if it is not.
        '''
        if ('counters' not in start_config or
            'noise' not in start_config or
            'noise_weight' not in start_config or
            'dc_threshold' not in start_config or
            'collect_period' not in start_config):
            logging.warning("start command from tally server cannot be completed due to missing data")
            return None

        # a traffic model is optional
        if 'traffic_model' in start_config:
            # if a traffic model was given but is not valid, fail
            if not check_traffic_model_config(start_config['traffic_model']):
                return None

            # regsiter the dependencies for the dynamic counter labels
            TrafficModel(start_config['traffic_model']).register_counters()

        # if the counters don't pass the validity checks, fail
        if not check_counters_config(start_config['counters'],
                                     start_config['noise']['counters']):
            return None

        # if the noise weights don't pass the validity checks, fail
        if not check_noise_weight_config(start_config['noise_weight'],
                                         start_config['dc_threshold']):
            return None

        # check if we need to delay this round
        if not self.collection_delay.round_start_permitted(
            start_config['noise'],
            time(),
            self.config['delay_period'],
            self.config['always_delay'],
            self.config['sigma_decrease_tolerance']):
            # we can't start the round yet
            return None

        # save various config items for the end of the round
        self.set_round_start(start_config)

        # combine bins and sigmas
        return combine_counters(start_config['counters'],
                                start_config['noise']['counters'])

    def check_stop_config(self, stop_config, counts):
        '''
        When the round stops, perform common client actions:
        - log a message
        - tell the collection_delay
        '''
        end_time = time()
        response = {}
        round_successful = False

        wants_counters = stop_config.get('send_counters', False)
        logging.info("tally server {} final counts"
                     .format("wants" if wants_counters else "does not want"))

        if wants_counters and counts is not None:
            logging.info("sending counts from {} counters".format(len(counts)))
            response['Counts'] = counts
            # only delay a round if we have sent our counters
            round_successful = True
        else:
            logging.info("No counts available")

        # even though the counter limits are hard-coded, include them anyway
        response['Config'] = add_counter_limits_to_config(self.config)

        # and include the config sent by the tally server in do_start
        if self.start_config is not None:
            response['Config']['Start'] = self.start_config

        # and include the config sent by the tally server to stop
        if stop_config is not None:
            response['Config']['Stop'] = stop_config

        # if we never started, there's no point in registering end of round
        if (self.collect_period is None or
            self.last_noise_config is None or
            self.collection_start_time is None):
            logging.warning("TS sent stop command before start command")
            return response

        # Using the collect_period from the tally server is inaccurate,
        # because the DCs and SKs do not check that the actual collection time
        # requested by the TS matches the collection period it claims
        actual_collect = end_time - self.collection_start_time

        # add this info to the context
        response['Config']['Time'] = {}
        response['Config']['Time']['Start'] = self.collection_start_time
        response['Config']['Time']['Stop'] = end_time
        response['Config']['Time']['Collect'] = actual_collect

        # Register the stop with the collection delay
        self.collection_delay.set_stop_result(
            round_successful,
            # set when the round started
            self.last_noise_config,
            self.collection_start_time,
            end_time,
            self.config['delay_period'],
            self.config['always_delay'],
            self.config['sigma_decrease_tolerance'])

        logging.info("collection phase was stopped")

        return response
