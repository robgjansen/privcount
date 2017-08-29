'''
Created on Dec 12, 2015

@author: rob

See LICENSE for licensing information
'''
import os
import json
import logging
import cPickle as pickle
import yaml

from time import time
from copy import copy, deepcopy
from base64 import b64encode

from twisted.internet import reactor, task, ssl
from twisted.internet.protocol import ServerFactory

from privcount.config import normalise_path, choose_secret_handshake_path
from privcount.counter import SecureCounters, counter_modulus, min_blinded_counter_value, max_blinded_counter_value, min_tally_counter_value, max_tally_counter_value, add_counter_limits_to_config, check_noise_weight_config, check_counters_config, CollectionDelay, float_accuracy, count_bins, are_events_expected
from privcount.crypto import generate_keypair, generate_cert
from privcount.log import log_error, format_elapsed_time_since, format_elapsed_time_wait, format_delay_time_until, format_interval_time_between, format_last_event_time_since, errorCallback, summarise_string
from privcount.node import PrivCountServer, continue_collecting, log_tally_server_status, EXPECTED_EVENT_INTERVAL_MAX, EXPECTED_CONTROL_ESTABLISH_MAX
from privcount.protocol import PrivCountServerProtocol, get_privcount_version
from privcount.statistics_noise import get_noise_allocation, get_sanity_check_counter, DEFAULT_DUMMY_COUNTER_NAME
from privcount.traffic_model import TrafficModel, check_traffic_model_config

# for warning about logging function and format # pylint: disable=W1202
# for calling methods on reactor # pylint: disable=E1101

class TallyServer(ServerFactory, PrivCountServer):
    '''
    receive blinded counts from the DCs
    receive key shares from the SKs
    sum shares and counts at end of epoch
    publish the final results to a file
    '''

    def __init__(self, config_filepath):
        PrivCountServer.__init__(self, config_filepath)
        self.clients = {}
        self.collection_phase = None
        self.idle_time = time()
        self.num_completed_collection_phases = 0
        self.refresh_task = None

    def buildProtocol(self, addr):
        '''
        Called by twisted
        '''
        return PrivCountServerProtocol(self)

    def startFactory(self):
        '''
        Called by twisted
        '''
        # TODO
        return
        state = self.load_state()
        if state is not None:
            self.clients = state['clients']
            self.collection_phase = state['collection_phase']
            self.idle_time = state['idle_time']

    def stopFactory(self):
        # TODO
        if self.refresh_task is not None and self.refresh_task.running:
            self.refresh_task.stop()
            self.refresh_task = None
        return
        if self.collection_phase is not None or len(self.clients) > 0:
            # export everything that would be needed to survive an app restart
            state = {'clients': self.clients, 'collection_phase': self.collection_phase, 'idle_time': self.idle_time}
            self.dump_state(state)

    def run(self):
        '''
        Called by twisted
        '''
        # load initial config
        self.refresh_config()
        if self.config is None:
            logging.critical("cannot start due to error in config file")
            return

        # refresh and check status every event_period seconds
        self.refresh_task = task.LoopingCall(self.refresh_loop)
        refresh_deferred = self.refresh_task.start(self.config['event_period'], now=False)
        refresh_deferred.addErrback(errorCallback)

        # setup server for receiving blinded counts from the DC nodes and key shares from the SK nodes
        listen_port = self.config['listen_port']
        key_path = self.config['key']
        cert_path = self.config['cert']
        ssl_context = ssl.DefaultOpenSSLContextFactory(key_path, cert_path)

        logging.info("Tally Server listening on port {}".format(listen_port))
        reactor.listenSSL(listen_port, self, ssl_context)
        reactor.run()

    def refresh_loop(self):
        '''
        Perform the TS event loop:
        Refresh the config, check clients, check if we want to start or stop
        collecting, and log a status update.
        This function is called using LoopingCall, so any exceptions will be
        turned into log messages. (This is the behaviour we want for malformed
        config files.)
        '''
        # make sure we have the latest config and counters
        self.refresh_config()

        # check if any clients have not checked in recently
        self.clear_dead_clients()

        # check if we should start the next collection phase
        if self.collection_phase is None:
            num_phases = self.num_completed_collection_phases
            if continue_collecting(num_phases,
                                   self.config['continue'],
                                   'idle'):
                dcs, sks = self.get_idle_dcs(), self.get_idle_sks()
                if len(dcs) >= self.config['dc_threshold'] and len(sks) >= self.config['sk_threshold']:
                    if self.collection_delay.round_start_permitted(
                            self.config['noise'],
                            time(),
                            self.config['delay_period'],
                            self.config['always_delay'],
                            self.config['sigma_decrease_tolerance']):
                        # we've passed all the checks, start the collection
                        num_phases = self.num_completed_collection_phases
                        logging.info("starting collection phase {} with {} DataCollectors and {} ShareKeepers".format((num_phases+1), len(dcs), len(sks)))
                        self.start_new_collection_phase(dcs, sks)

        # check if we should stop a running collection phase
        else:
            if self.collection_phase.is_error():
                logging.info("stopping collection phase due to error")
                self.stop_collection_phase()

            elif self.collection_phase.is_expired():
                logging.info("stopping collection phase due to valid expiration")
                self.stop_collection_phase()

        # log the latest status
        log_tally_server_status(self.get_status())
        if self.collection_phase is not None:
            self.collection_phase.log_status()

    def refresh_config(self):
        '''
        re-read config and process any changes
        '''
        # TODO: refactor common code: see ticket #121
        try:
            logging.debug("reading config file from '%s'", self.config_filepath)

            # read in the config from the given path
            with open(self.config_filepath, 'r') as fin:
                conf = yaml.load(fin)
            ts_conf = conf['tally_server']

            # a private/public key pair and a cert containing the public key
            # if either path is not specified, use the default path
            if 'key' in ts_conf and 'cert' in ts_conf:
                ts_conf['key'] = normalise_path(ts_conf['key'])
                ts_conf['cert'] = normalise_path(ts_conf['cert'])
            else:
                ts_conf['key'] = normalise_path('privcount.rsa_key.pem')
                ts_conf['cert'] = normalise_path('privcount.rsa_key.cert')
            # generate a new key and cert if either file does not exist
            if (not os.path.exists(ts_conf['key']) or
                not os.path.exists(ts_conf['cert'])):
                generate_keypair(ts_conf['key'])
                generate_cert(ts_conf['key'], ts_conf['cert'])

            # find the path for the secret handshake file
            ts_conf['secret_handshake'] = choose_secret_handshake_path(
                ts_conf, conf)
            # check we can load the secret handshake file, creating if needed
            # (but ignore the actual secret, it never forms part of our config)
            # we can't use PrivCountProtocol.handshake_secret(), because
            # our own self.config is either None or outdated at this point
            assert PrivCountServerProtocol.handshake_secret_load(
                ts_conf['secret_handshake'],
                create=True)

            ts_conf.setdefault('circuit_sample_rate', 1.0)
            ts_conf['circuit_sample_rate'] = \
                float(ts_conf['circuit_sample_rate'])
            assert ts_conf['circuit_sample_rate'] >= 0.0
            assert ts_conf['circuit_sample_rate'] <= 1.0

            ts_conf.setdefault('max_cell_events_per_circuit', -1)
            ts_conf['max_cell_events_per_circuit'] = \
                int(ts_conf['max_cell_events_per_circuit'])

            # the counter bin file
            if 'counters' in ts_conf:
                ts_conf['counters'] = normalise_path(ts_conf['counters'])
                assert os.path.exists(ts_conf['counters'])
                with open(ts_conf['counters'], 'r') as fin:
                    counters_conf = yaml.load(fin)
                ts_conf['counters'] = counters_conf['counters']
            else:
                ts_conf['counters'] = conf['counters']

            # the counter noise config - one of the following must be provided:
            # noise: contains the noise allocation parameters,
            # sigmas: contains pre-calculated sigmas
            # (if both are provided, sigmas is ignored)
            # noise file
            if 'noise' in ts_conf:
                ts_conf['noise'] = normalise_path(ts_conf['noise'])
                assert os.path.exists(ts_conf['noise'])
                with open(ts_conf['noise'], 'r') as fin:
                    noise_conf = yaml.load(fin)
                # use both the privacy and counters elements from noise_conf
                ts_conf['noise'] = {}
                ts_conf['noise']['privacy'] = noise_conf['privacy']
                ts_conf['noise']['counters'] = noise_conf['counters']
            # noise config in the same file
            elif 'privacy' in conf and 'counters' in conf:
                ts_conf['noise'] = {}
                ts_conf['noise']['privacy'] = conf['privacy']
                ts_conf['noise']['counters'] = conf['counters']
            # sigmas file
            elif 'sigmas' in ts_conf:
                ts_conf['sigmas'] = normalise_path(ts_conf['sigmas'])
                assert os.path.exists(ts_conf['sigmas'])
                with open(ts_conf['sigmas'], 'r') as fin:
                    sigmas_conf = yaml.load(fin)
                ts_conf['noise'] = {}
                ts_conf['noise']['counters'] = sigmas_conf['counters']
                # we've packed it into ts_conf['noise'], so remove it
                del ts_conf['sigmas']
            # sigmas config in the same file
            else:
                ts_conf['noise'] = {}
                ts_conf['noise']['counters'] = conf['counters']

            # if we are counting a traffic model
            if 'traffic_model' in ts_conf:
                # we need the model, which specifies which counters we need to count
                # make sure the model file exists
                ts_conf['traffic_model'] = normalise_path(ts_conf['traffic_model'])
                assert os.path.exists(ts_conf['traffic_model'])

                # import and validate the model
                with open(ts_conf['traffic_model'], 'r') as fin:
                    traffic_model_conf = json.load(fin)
                    assert check_traffic_model_config(traffic_model_conf)

                # store the configs so we can transfer them later
                ts_conf['traffic_model'] = traffic_model_conf

                # get an object and register the dynamic counters
                tmodel = TrafficModel(traffic_model_conf)

                # we also need noise parameters for all of the traffic model counters
                # make sure the noise file exists
                assert 'traffic_noise' in ts_conf
                ts_conf['traffic_noise'] = normalise_path(ts_conf['traffic_noise'])
                assert os.path.exists(ts_conf['traffic_noise'])

                # import and validate the noise
                with open(ts_conf['traffic_noise'], 'r') as fin:
                    traffic_noise_conf = yaml.load(fin)
                assert tmodel.check_noise_config(traffic_noise_conf)

                # store the configs so we can transfer them later
                ts_conf['traffic_noise'] = traffic_noise_conf

                # supplying a traffic model implies that the tally server
                # wants to enable all counters associated with that model
                # register the dynamic counter labels that will be needed for this model
                tmodel.register_counters()

                # get the bins and noise that we should use for this model
                tmodel_bins = tmodel.get_bins_init_config()
                tmodel_noise = tmodel.get_noise_init_config(traffic_noise_conf)

                # sanity check
                if set(tmodel_bins.keys()) != set(tmodel_noise.keys()):
                    logging.error("the set of initial bins and noise labels are not equal")
                    assert False

                # inject the traffic model counter bins and noise configs, i.e.,
                # append the traffic model bins and noise to the other configured values
                ts_conf['counters'].update(tmodel_bins)
                ts_conf['noise']['counters'].update(tmodel_noise)

            # an optional noise allocation results file
            if 'allocation' in ts_conf:
                ts_conf['allocation'] = normalise_path(ts_conf['allocation'])
                assert os.path.exists(os.path.dirname(ts_conf['allocation']))

            # now all the files are loaded, use noise to calculate sigmas
            # (if noise was configured)
            if 'privacy' in ts_conf['noise']:
                ts_conf['noise'] = get_noise_allocation(
                    ts_conf['noise'],
                    circuit_sample_rate=ts_conf['circuit_sample_rate'])
                # and write it to the specified file (if configured)
                if 'allocation' in ts_conf:
                    with open(ts_conf['allocation'], 'w') as fout:
                        yaml.dump(ts_conf['noise'], fout,
                                  default_flow_style=False)

            # ensure we always add a sanity check counter
            ts_conf['counters'][DEFAULT_DUMMY_COUNTER_NAME] = get_sanity_check_counter()
            ts_conf['noise']['counters'][DEFAULT_DUMMY_COUNTER_NAME] = get_sanity_check_counter()

            # now we have bins and sigmas (and perhaps additional calculation
            # info along with the sigmas)
            # perform sanity checks, making sure all counter names are known
            # counters
            assert check_counters_config(ts_conf['counters'],
                                         ts_conf['noise']['counters'],
                                         allow_unknown_counters=False)

            # a directory for results files
            if 'results' in ts_conf:
                ts_conf['results'] = normalise_path(ts_conf['results'])
            else:
                ts_conf['results'] = normalise_path('./')
            assert os.path.exists(ts_conf['results'])

            # the state file (unused)
            if 'state' in ts_conf:
                del ts_conf['state']
            #ts_conf['state'] = normalise_path(ts_conf['state'])
            #assert os.path.exists(os.path.dirname(ts_conf['state']))

            # Must be configured manually
            assert 'collect_period' in ts_conf
            # Set the default periods
            ts_conf.setdefault('event_period', 60)
            ts_conf.setdefault('checkin_period', 60)

            # The event period should be less than or equal to half the
            # collect period, otherwise privcount sometimes takes an extra
            # event period to produce results
            event_max = ts_conf['collect_period']/2
            if (ts_conf['event_period'] > event_max):
                logging.warning("event_period %d too large for collect_period %d, reducing to %d",
                                ts_conf['event_period'],
                                ts_conf['collect_period'],
                                event_max)
                ts_conf['event_period'] = event_max

            # The checkin period must be less than or equal to half the
            # collect period, otherwise privcount never finishes.
            checkin_max = ts_conf['collect_period']/2
            if (ts_conf['checkin_period'] > checkin_max):
                logging.warning("checkin_period %d too large for collect_period %d, reducing to %d",
                                ts_conf['checkin_period'],
                                ts_conf['collect_period'],
                                checkin_max)
                ts_conf['checkin_period'] = checkin_max
            # It should also be less than or equal to the event period,
            # so that the TS is up to date with client statuses every
            # event loop.
            checkin_max_log = ts_conf['event_period']
            if (ts_conf['checkin_period'] > checkin_max_log):
                logging.info("checkin_period %d greater than event_period %d, client statuses might be delayed",
                             ts_conf['checkin_period'],
                             ts_conf['event_period'])

            ts_conf['delay_period'] = self.get_valid_delay_period(ts_conf)

            ts_conf.setdefault('always_delay', False)
            assert isinstance(ts_conf['always_delay'], bool)

            ts_conf['sigma_decrease_tolerance'] = \
                self.get_valid_sigma_decrease_tolerance(ts_conf)

            assert ts_conf['listen_port'] > 0
            assert ts_conf['sk_threshold'] > 0
            assert ts_conf['dc_threshold'] > 0
            assert ts_conf.has_key('noise_weight')
            assert check_noise_weight_config(ts_conf['noise_weight'],
                                             ts_conf['dc_threshold'])
            assert ts_conf['collect_period'] > 0
            assert ts_conf['event_period'] > 0
            assert ts_conf['checkin_period'] > 0
            # The TS runs one round by default
            ts_conf.setdefault('continue', False)
            assert (isinstance(ts_conf['continue'], bool) or
                    ts_conf['continue'] >= 0)
            # check the hard-coded counter values are sane
            assert counter_modulus() > 0
            assert min_blinded_counter_value() == 0
            assert max_blinded_counter_value() > 0
            assert max_blinded_counter_value() < counter_modulus()
            assert min_tally_counter_value() < 0
            assert max_tally_counter_value() > 0
            assert max_tally_counter_value() < counter_modulus()
            assert -min_tally_counter_value() < counter_modulus()

            if self.config == None:
                self.config = ts_conf
                logging.info("using config = %s",
                             summarise_string(str(self.config), 100))
                logging.debug("using config (full value) = %s",
                              str(self.config))
            else:
                changed = False
                for k in ts_conf:
                    if k not in self.config or ts_conf[k] != self.config[k]:
                        old_val_str = str(self.config[k]) if k in self.config else '(absent)'
                        new_val_str = str(ts_conf[k])
                        logging.info("updated config for key {} from {} to {}"
                                     .format(k,
                                             summarise_string(
                                                          old_val_str,
                                                          100),
                                             summarise_string(
                                                          new_val_str,
                                                          100)))
                        logging.debug("updated config for key {} (full values) from {} to {}"
                                      .format(k, self.config[k], ts_conf[k]))
                        self.config[k] = ts_conf[k]
                        changed = True
                if not changed:
                    logging.debug('no config changes found')

        except AssertionError:
            logging.warning("problem reading config file: invalid data")
            log_error()
        except KeyError:
            logging.warning("problem reading config file: missing required keys")
            log_error()

    def get_max_client_rtt(self, uid):
        '''
        Get the maximum reasonable rtt for uid
        '''
        # Maximum RTT in ~2005 was 20 seconds
        # https://www3.cs.stonybrook.edu/~phillipa/papers/SPECTS.pdf
        # There's no guarantee the last rtt will be the same as this one,
        # so add a few seconds unconditionally
        return self.clients[uid].get('rtt', 15.0) + 5.0

    def is_client_control_ok(self, uid):
        '''
        Has uid completed the control protocol with its tor instance within a
        reasonable amount of time, taking into account checkin period, rtt,
        collection phase start time, and clock padding?
        '''
        now = time()
        c_status = self.clients[uid]
        if c_status['type'] != 'DataCollector':
            return True

        # if the collection phase hasn't started, everything is ok
        if self.collection_phase is None:
            return True
        start_ts = self.collection_phase.get_start_ts()
        if start_ts is None:
            return True

        # if we've completed the control protocol, everything is ok
        if 'tor_privcount_version' in c_status:
            return True

        rtt = self.get_max_client_rtt(uid)
        clock_padding = self.collection_phase.clock_padding
        time_since_start = now - (start_ts + clock_padding)

        # This will also trigger if we miss a checkin at the start of the
        # round. That's ok.
        return time_since_start <= (EXPECTED_CONTROL_ESTABLISH_MAX +
                                    self.get_checkin_period() +
                                    rtt)

    def is_last_client_event_recent(self, uid):
        '''
        Is the last event from uid newer than EXPECTED_EVENT_INTERVAL_MAX,
        taking into account the checkin period, rtt, the collection phase
        start time, and clock padding?
        '''
        now = time()
        c_status = self.clients[uid]
        if c_status['type'] != 'DataCollector':
            return True

        # if the collection phase hasn't started, everything is ok
        if self.collection_phase is None:
            return True
        start_ts = self.collection_phase.get_start_ts()
        if start_ts is None:
            return True

        rtt = self.get_max_client_rtt(uid)
        clock_padding = self.collection_phase.clock_padding
        time_since_event = now - c_status.get('last_event_time',
                                              start_ts + clock_padding)

        return time_since_event <= (EXPECTED_EVENT_INTERVAL_MAX +
                                    2*self.get_checkin_period() +
                                    rtt)

    def are_dc_events_expected(self, uid, status=None):
        '''
        Return True if we expect the Data Collector at uid to receive events
        regularly.
        Return False if we don't, or if it's not a Data Collector.
        '''
        if self.get_client_type(uid, status) != 'DataCollector':
            return False

        flag_list = self.get_client_flag_list(uid, status)
        return are_events_expected(self.config['counters'], flag_list)

    def clear_dead_clients(self):
        '''
        Check how long it has been since clients have successfully contacted
        us, and mark clients that have been down for too long.
        Also warns if clients have been various kinds of down for a smaller
        amount of time.
        '''
        now = time()

        for uid in self.clients.keys():
            # don't print ShareKeepers' public keys, they're very long
            c_status = self.clients[uid].copy()
            if 'public_key' in c_status:
                c_status['public_key'] = "(public key)"
            time_since_checkin = now - c_status['alive']
            start_ts = None
            if self.collection_phase is not None:
                start_ts = self.collection_phase.get_start_ts()
            time_since_event = 0.0
            if c_status['type'] == 'DataCollector' and start_ts:
                time_since_event = now - c_status.get('last_event_time',
                                                      start_ts)
            rtt = self.get_max_client_rtt(uid)

            flag_message = "Is the relay in the Tor consensus?"
            flag_list = self.get_client_flag_list(uid)
            if flag_list is None:
                flag_message = ""
            elif len(flag_list) > 0:
                flag_message = "Consensus flags: {}".format(" ".join(flag_list))

            if self.are_dc_events_expected(uid):
                log_fn = logging.warning
            else:
                log_fn = logging.info

            cname = TallyServer.get_client_display_name(uid)
            cdetail = self.get_client_detail(uid)

            if not self.is_client_control_ok(uid):
                logging.warning("control connection delayed more than {}s for client {} {}"
                                .format(EXPECTED_CONTROL_ESTABLISH_MAX,
                                        cname, c_status))

            if not self.is_last_client_event_recent(uid):
                log_fn("{} for client {} {} {}"
                       .format(format_last_event_time_since(
                                            c_status.get('last_event_time')),
                               cname, c_status, flag_message))

            if time_since_checkin > 3 * self.get_checkin_period() + rtt:
                logging.warning("last checkin was {} for client {} {}"
                                .format(format_elapsed_time_wait(
                                            time_since_checkin, 'at'),
                                        cname, c_status))

            if time_since_checkin > 7 * self.get_checkin_period() + rtt:
                logging.warning("marking dead client {} {}"
                                .format(cname, cdetail))
                c_status['state'] = 'dead'

                if self.collection_phase is not None and self.collection_phase.is_participating(uid):
                    self.collection_phase.lost_client(uid)

                self.clients.pop(uid, None)

    def _get_matching_clients(self, c_type, c_state, c_key=None):
        matching_clients = []
        for uid in self.clients:
            if (self.clients[uid]['type'] == c_type and
                self.clients[uid]['state'] == c_state and
                (c_key is None or c_key in self.clients[uid])):
                matching_clients.append(uid)
        return matching_clients

    def get_idle_dcs(self):
        return self._get_matching_clients('DataCollector', 'idle')

    def get_active_dcs(self):
        return self._get_matching_clients('DataCollector', 'active')

    def get_control_dcs(self):
        '''
        Return the set of DCs that have successfully controlled a tor process.
        This does *not* use is_client_control_ok().
        '''
        return self._get_matching_clients('DataCollector', 'active',
                                          'tor_privcount_version')

    def get_event_dcs(self):
        '''
        Return the set of DCs that have received an event recently.
        See is_last_client_event_recent() for details.
        '''
        matching_clients = []
        control_dcs = self.get_control_dcs()
        for uid in control_dcs:
            if self.is_last_client_event_recent(uid):
                matching_clients.append(uid)
        return matching_clients

    def get_idle_sks(self):
        return self._get_matching_clients('ShareKeeper', 'idle')

    def get_active_sks(self):
        return self._get_matching_clients('ShareKeeper', 'active')

    def count_client_states(self):
        dc_idle = len(self.get_idle_dcs())
        dc_active = len(self.get_active_dcs())
        sk_idle = len(self.get_idle_sks())
        sk_active = len(self.get_active_sks())
        return dc_idle, dc_active, sk_idle, sk_active

    def get_checkin_period(self): # called by protocol
        return self.config['checkin_period']

    def get_status(self): # called by protocol
        dc_idle, dc_active, sk_idle, sk_active = self.count_client_states()

        status = {
            'state' : 'idle' if self.collection_phase is None else 'active',
            'time' : self.idle_time if self.collection_phase is None else self.collection_phase.get_start_ts(),
            'dcs_idle' : dc_idle,
            'dcs_active' : dc_active,
            'dcs_total' : dc_idle+dc_active,
            'dcs_required' : self.config['dc_threshold'],
            'dcs_control' : len(self.get_control_dcs()),
            'dcs_event' : len(self.get_event_dcs()),
            'sks_idle' : sk_idle,
            'sks_active' : sk_active,
            'sks_total' : sk_idle+sk_active,
            'sks_required' : self.config['sk_threshold'],
            'completed_phases' : self.num_completed_collection_phases,
            'continue' : self.config['continue'],
            'delay_until' : self.collection_delay.get_next_round_start_time(
                self.config['noise'],
                self.config['delay_period'],
                self.config['always_delay'],
                self.config['sigma_decrease_tolerance']),
            'privcount_version' : get_privcount_version(),
        }

        # we can't know the expected end time until we have started
        if self.collection_phase is not None:
            starting_ts = self.collection_phase.get_start_ts()
            if starting_ts is not None:
                status['expected_end_time'] = starting_ts + self.config['collect_period']

        return status

    def _get_client_item(self, uid, item, status=None, substitute=None):
        '''
        Tries to find item in status, or, if status is None, tries
        self.clients[uid].
        Returns substitute if there is no item.
        '''
        assert uid is not None

        if status is None:
            status = self.clients[uid]

        return status.get(item, substitute)

    @staticmethod
    def get_client_display_name(uid):
        '''
        Returns a display name, based on uid, that is a suitable length for
        logging.
        '''
        # Allow standard-length tor relay nicknames and fingerprints
        # Replace entire hex characters when summarising, not just ...
        return summarise_string(uid, 20, ellipsis='....')

    def get_client_type(self, uid, status=None):
        '''
        Uses _get_client_item to find the client type for uid.
        Returns None if client does not have a type.
        '''
        return self._get_client_item(uid,
                                     'type',
                                     status,
                                     None)

    def get_client_address(self, uid, status=None):
        '''
        Uses _get_client_item to find the remote peer info (hostname and port)
        for uid.
        Returns a placeholder string if client does not have an address.
        '''
        return self._get_client_item(uid,
                                     'client_address',
                                     status,
                                     '(no remote address)')

    def get_client_nickname(self, uid, status=None):
        '''
        Uses _get_client_item to find a fingerprint.
        Returns None if client will never have a nickname, and placeholder
        strings if we know it has no nickname, or we expect a nickname in
        future.
        '''
        if self.get_client_type(uid, status) != 'DataCollector':
            return None

        nick = self._get_client_item(uid,
                                     'nickname',
                                     status,
                                     '(nickname pending)')
        # Distinguish between unknown and known empty nicknames
        if len(nick) == 0:
            nick = '(no nickname)'
        return nick

    def get_client_fingerprint(self, uid, status=None):
        '''
        Uses _get_client_item to find a fingerprint.
        Returns None if client will never have a fingerprint, and a
        placeholder string if we expect a fingerprint in future.
        '''
        if self.get_client_type(uid, status) != 'DataCollector':
            return None

        return self._get_client_item(uid,
                                     'fingerprint',
                                     status,
                                     '(fingerprint pending)')

    def get_client_flag_list(self, uid, status=None):
        '''
        Return the flags for uid in latest status (updated from its latest
        consensus).
        If there are no flags, return an empty list.
        If it's not a Data Collector, return None.
        '''
        if self.get_client_type(uid, status) != 'DataCollector':
            return None

        return self._get_client_item(uid,
                                     'flag_list',
                                     status,
                                     [])

    def get_client_info(self, uid, status=None):
        '''
        Returns a formatted string containing basic information: the
        client's address (if present).
        '''
        return self.get_client_address(uid, status)

    def get_client_detail(self, uid, status=None):
        '''
        Returns a formatted string containing detailed information: the
        client's nickname, address, and fingerprint (if present).
        '''
        if self.get_client_type(uid, status) != 'DataCollector':
            return self.get_client_info(uid, status)

        return "{} {} {}".format(self.get_client_nickname(uid, status),
                                 self.get_client_address(uid, status),
                                 self.get_client_fingerprint(uid, status))

    def get_client_version(self, uid, status=None):
        '''
        Uses _get_client_item to find privcount_version, tor_version, and
        tor_privcount_version.
        Returns a formatted string containing the versions that are present.
        '''
        privcount_version = self._get_client_item(uid,
                                                  'privcount_version',
                                                  status,
                                                  '(no privcount version)')

        # if we're not expecting additional versions, just use privcount
        if self.get_client_type(uid, status) != 'DataCollector':
            return privcount_version

        tor_version = self._get_client_item(uid,
                                            'tor_version',
                                            status,
                                            '(pending)')
        tor_privcount_version = self._get_client_item(uid,
                                                      'tor_privcount_version',
                                                      status,
                                                      '(pending)')

        return ('privcount: {} tor: {} tor privcount: {}'
                .format(privcount_version, tor_version, tor_privcount_version))

    def is_valid_client_version(self, uid, status=None):
        '''
        Check that the version of client is new enough that we want to use it.
        Warn and return False if it is not.
        '''
        cname = TallyServer.get_client_display_name(uid)
        cinfo = self.get_client_info(uid, status)
        cdetail = self.get_client_detail(uid, status)
        cversion = self.get_client_version(uid, status)

        # Reject DC versions 1.0.0 and 1.0.1, they didn't add noise
        client_type = self.get_client_type(uid, status)
        pc_version = self._get_client_item(uid, 'privcount_version',
                                                  status, None)
        pc_version_number, _, _ = pc_version.partition(' ')
        if client_type == 'DataCollector':
            if pc_version_number == '1.0.0' or pc_version_number == '1.0.1':
                logging.warning("Insecure Data Collector PrivCount version {}: {} {}"
                                .format(pc_version_number, cname, cinfo))
                logging.debug("Insecure Data Collector PrivCount version {}: {} detail {} {}"
                              .format(pc_version_number, cname, cdetail,
                                      cversion))
                return False

        return True

    def set_client_status(self, uid, status): # called by protocol
        cname = TallyServer.get_client_display_name(uid)
        cinfo = self.get_client_info(uid, status)
        cdetail = self.get_client_detail(uid, status)
        cversion = self.get_client_version(uid, status)

        # dump the status content at debug level
        logging.debug("{} {} sent status: {}"
                      .format(cname, cdetail, status))
        if uid in self.clients:
            logging.debug("{} {} has stored state: {}"
                          .format(cname, cdetail, self.clients[uid]))

        # Warn and ignore invalid clients
        if not self.is_valid_client_version(uid, status):
            return

        # only data collectors have a fingerprint
        # oldfingerprint is the previous fingerprint for this client (if any)
        # fingerprint is the current fingerprint in the status (if any)
        # If there is no fingerprint, these are None
        oldfingerprint = self.clients.get(uid, {}).get('fingerprint')
        fingerprint = status.get('fingerprint', None)

        # complain if fingerprint changes, and keep the old one
        if (fingerprint is not None and oldfingerprint is not None and
            fingerprint != oldfingerprint):
            logging.warning("Ignoring fingerprint update from {} {} {} (version: {}) state {}: kept old {} ignored new {}"
                            .format(status['type'], cname, cinfo, cversion,
                                    status['state'], oldfingerprint, fingerprint))

        if uid not in self.clients:
            # data collectors don't have nickname, fingerprint, or tor
            # versions until the round starts
            logging.info("new {} {} {} joined and is {}"
                         .format(status['type'], cname, cinfo,
                                 status['state']))

        oldstate = self.clients[uid]['state'] if uid in self.clients else status['state']
        # for each key, replace the client value with the value from status,
        # or, if uid is a new client, initialise uid with status
        self.clients.setdefault(uid, status).update(status)
        # use status['alive'] as the initial value of 'time'
        self.clients[uid].setdefault('time', status['alive'])
        if oldstate != self.clients[uid]['state']:
            self.clients[uid]['time'] = status['alive']
        # always keep the old fingerprint
        if oldfingerprint is not None:
            self.clients[uid]['fingerprint'] = oldfingerprint

        last_event_time = status.get('last_event_time', None)
        last_event_message = ""
        # only log a message if we expect events
        if self.clients[uid]['type'] == 'DataCollector':
            last_event_message = ' ' + format_last_event_time_since(last_event_time)
        logging.info("----client status: {} {} is alive and {} for {}{}"
                     .format(self.clients[uid]['type'], cname,
                             self.clients[uid]['state'],
                             format_elapsed_time_since(self.clients[uid]['time'], 'since'),
                             last_event_message))
        logging.info("----client status: {} detail {} {} version: {}"
                     .format(self.clients[uid]['type'],
                             uid,
                             cdetail,
                             cversion))

    def get_clock_padding(self, client_uids):
        max_delay = max([self.clients[uid]['rtt']+self.clients[uid]['clock_skew'] for uid in client_uids])
        return max_delay + self.get_checkin_period()

    def start_new_collection_phase(self, dc_uids, sk_uids):
        assert self.collection_phase is None

        clock_padding = self.get_clock_padding(dc_uids + sk_uids)

        sk_public_keys = {}
        for uid in sk_uids:
            sk_public_keys[uid] = self.clients[uid]['public_key']

        traffic_model_conf = None
        if 'traffic_model' in self.config:
            traffic_model_conf = self.config['traffic_model']

        # clients don't provide some context until the end of the phase
        # so we'll wait and pass the client context to collection_phase just
        # before stopping it

        self.collection_phase = CollectionPhase(self.config['collect_period'],
                                                self.config['counters'],
                                                traffic_model_conf,
                                                self.config['noise'],
                                                self.config['noise_weight'],
                                                self.config['dc_threshold'],
                                                sk_uids,
                                                sk_public_keys,
                                                dc_uids,
                                                counter_modulus(),
                                                clock_padding,
                                                self.config['max_cell_events_per_circuit'],
                                                self.config['circuit_sample_rate'],
                                                self.config)
        self.collection_phase.start()

    def stop_collection_phase(self):
        assert self.collection_phase is not None
        self.collection_phase.set_client_status(self.clients)
        self.collection_phase.set_tally_server_status(self.get_status())
        self.collection_phase.stop()
        if self.collection_phase.is_stopped():
            # warn the user if a DC didn't collect any data
            for uid in self.clients:
                if self.clients[uid]['type'] == 'DataCollector':
                    last_event = self.clients[uid].get('last_event_time', None)
                    event_issue = None
                    if last_event is None:
                        event_issue = 'never received any events'
                    elif last_event < self.collection_phase.get_start_ts():
                        event_issue = 'received an event before collection started'
                    if event_issue is not None:
                        cname = TallyServer.get_client_display_name(uid)
                        cdetail = self.get_client_detail(uid)
                        # we could refuse to provide any results here, but they
                        # could still be useful even if they are missing a DC
                        # (or the DC has clock skew). So deliver the results
                        # and allow the operator to decide how to interpret
                        # them
                        logging.warning('Data Collector {} {} {}. Check the results before using them.'
                                        .format(cname, cdetail, event_issue))

            # we want the end time after all clients have definitely stopped
            # and returned their results, not the time the TS told the
            # CollectionPhase to initiate the stop procedure
            # (otherwise, a lost message or down client could delay stopping
            # for a pathological period of time, breaking our assumptions)
            # This also means that the SKs will allow the round to start
            # slightly before the TS allows it, which is a good thing.
            end_time = time()
            self.num_completed_collection_phases += 1
            self.collection_phase.write_results(self.config['results'],
                                                end_time)
            self.collection_delay.set_stop_result(
                not self.collection_phase.is_error(),
                # we can't use config['noise'], because it might have changed
                # since the start of the round
                self.collection_phase.get_noise_config(),
                self.collection_phase.get_start_ts(),
                end_time,
                # if config['delay_period'] has changed, we use it, and warn
                # if it would have made a difference
                self.config['delay_period'],
                self.config['always_delay'],
                self.config['sigma_decrease_tolerance'])
            self.collection_phase = None
            self.idle_time = time()

    def get_start_config(self, client_uid):
        '''
        called by protocol
        return None to indicate we shouldnt start the client yet
        '''
        if self.collection_phase is not None:
            return self.collection_phase.get_start_config(client_uid)
        else:
            return None

    def set_start_result(self, client_uid, result_data):
        '''
        called by protocol
        '''
        if self.collection_phase is not None:
            self.collection_phase.store_data(client_uid, result_data)

    def get_stop_config(self, client_uid):
        '''
        called by protocol
        returns None to indicate we shouldnt stop the client yet
        '''
        if self.collection_phase is not None:
            return self.collection_phase.get_stop_config(client_uid)
        elif client_uid in self.clients and self.clients[client_uid]['state'] == 'active':
            # client is active even though we have no collection phase (could be stale client)
            return {'send_counters' : False}
        else:
            return None

    def set_stop_result(self, client_uid, result_data): # called by protocol
        if self.collection_phase is not None:
            self.collection_phase.store_data(client_uid, result_data)

class CollectionPhase(object):

    def __init__(self, period, counters_config, traffic_model_config, noise_config,
                 noise_weight_config, dc_threshold_config, sk_uids,
                 sk_public_keys, dc_uids, modulus, clock_padding,
                 max_cell_events_per_circuit, circuit_sample_rate,
                 tally_server_config):
        # store configs
        self.period = period
        # the counter bins
        self.counters_config = counters_config
        self.traffic_model_config = traffic_model_config
        self.noise_config = noise_config
        self.noise_weight_config = noise_weight_config
        self.dc_threshold_config = dc_threshold_config
        self.sk_uids = sk_uids
        self.sk_public_keys = sk_public_keys
        self.dc_uids = dc_uids
        self.modulus = modulus
        self.clock_padding = clock_padding
        self.max_cell_events_per_circuit = max_cell_events_per_circuit
        self.circuit_sample_rate = circuit_sample_rate
        # make a deep copy, so we can delete unnecesary keys
        self.tally_server_config = deepcopy(tally_server_config)
        self.tally_server_status = None
        self.client_status = {}
        self.client_config = {}

        # setup some state
        self.state = 'new' # states: new -> starting_dcs -> starting_sks -> started -> stopping -> stopped
        self.starting_ts = None
        self.stopping_ts = None
        self.encrypted_shares = {} # uids of SKs to which we send shares {sk_uid : share_data}
        self.need_shares = set() # uids of DCs from which we still need encrypted shares
        self.final_counts = {} # uids of clients and their final reported counts
        self.need_counts = set() # uids of clients from which we still need final counts
        self.error_flag = False

    def _change_state(self, new_state):
        old_state = self.state
        self.state = new_state
        if old_state != new_state:
            logging.info("collection phase state changed from '{}' to '{}'".format(old_state, new_state))

    def start(self):
        if self.state != "new":
            return

        # we are now starting up
        self.starting_ts = time()

        # we first need to get all encrypted shares from the DCs before we
        # forward them to the SKs
        for uid in self.dc_uids:
            self.need_shares.add(uid)
        self._change_state('starting_dcs')

    def stop(self):
        if self.stopping_ts is None:
            self.stopping_ts = time()

        # main state switch to decide how to stop the phase
        if self.state == 'new':
            self._change_state('stopped')

        elif self.state == 'starting_dcs' or self.state == 'starting_sks':
            self.need_shares.clear()
            self.encrypted_shares.clear()

            # need to tell all clients to stop and reset
            self._change_state('stopping')
            for uid in self.dc_uids+self.sk_uids:
                self.need_counts.add(uid)
            self.error_flag = True # when sending STOP, indicate error so we dont get tallies

        elif self.state == 'started':
            # our noise covers activity independent of the length of the period
            # so we can keep results even if we are ending early
            if self.stopping_ts - self.starting_ts >= self.period:
                logging.info("graceful end to collection phase")
            else:
                logging.info("premature end to collection phase, results may be less accurate than expected due to the noise that was added (if a client is missing, results may be nonsense)")

            for uid in self.dc_uids+self.sk_uids:
                self.need_counts.add(uid)

            # when sending STOP, indicate that we need tallies
            self.error_flag = False
            self._change_state('stopping')

        elif self.state == 'stopping':
            if len(self.need_counts) == 0:
                self._change_state('stopped')

    def lost_client(self, client_uid):
        '''
        this is called when client_uid isn't responding
        we could mark error_flag as true and abort, or keep counting anyway
        and hope we can recover from the error by adding the local state
        files later... TODO
        '''
        pass

    def store_data(self, client_uid, data):
        cname = TallyServer.get_client_display_name(client_uid)

        if data == None:
            # this can happen if the SK (or DC) is enforcing a delay because
            # the noise allocation has changed
            logging.warning("received error response from {} while in state {}"
                            .format(cname, self.state))
            return

        if self.state == 'starting_dcs':
            # we expect these to be the encrpyted and blinded counts
            # from the DCs that we should forward to the SKs during SK startup
            assert client_uid in self.dc_uids

            # dont add a share from the same DC twice
            if client_uid in self.need_shares:
                # collect all shares for each SK together
                shares = data # dict of {sk_uid : share}
                for sk_uid in shares:
                    self.encrypted_shares.setdefault(sk_uid, []).append(shares[sk_uid])
                logging.info("received {} shares from data collector {}"
                             .format(len(shares), cname))

                # mark that we got another one
                self.need_shares.remove(client_uid)
                logging.info("need shares from {} more data collectors".format(len(self.need_shares)))
                if len(self.need_shares) == 0:
                    # ok, we got all of the shares for all SKs, now start the SKs
                    for sk_uid in self.sk_uids:
                        self.need_shares.add(sk_uid)
                    self._change_state('starting_sks')

        elif self.state == 'starting_sks':
            # the sk got our encrypted share successfully
            logging.info("share keeper {} started and received its shares"
                         .format(cname))
            self.need_shares.remove(client_uid)
            if len(self.need_shares) == 0:
                self._change_state('started')

        elif self.state == 'stopping':
            # record the configuration for the client context
            response_config = data.get('Config', None)
            if response_config is not None:
                self.set_client_config(client_uid, response_config)

            if client_uid in self.need_counts:
                # the client got our stop command
                counts = data.get('Counts', None)

                if counts is None:
                    logging.warning("received no counts from {}, final results will not be available"
                                    .format(cname))
                    self.error_flag = True
                elif not self.is_error() and len(counts) == 0:
                    logging.warning("received empty counts from {}, final results will not be available"
                                    .format(cname))
                    self.error_flag = True
                elif not self.is_error():
                    logging.info("received {} counters ({} bins) from stopped client {}"
                                 .format(len(counts), count_bins(counts),
                                         cname))
                    # add up the tallies from the client
                    self.final_counts[client_uid] = counts
                else:
                    logging.warning("received counts: error from stopped client {}"
                                    .format(cname))
                self.need_counts.remove(client_uid)

    def is_participating(self, client_uid):
        return True if client_uid in self.sk_uids or client_uid in self.dc_uids else False

    def is_expired(self):
        if self.starting_ts is None:
            return False
        return True if (time() - self.starting_ts) >= self.period else False

    def is_error(self):
        return self.error_flag

    def is_stopped(self):
        return True if self.state == 'stopped' else False

    def get_noise_config(self):
        return self.noise_config

    def get_start_ts(self):
        return self.starting_ts

    def get_start_config(self, client_uid):
        '''
        Get the starting DC or SK configuration.
        Called by protocol via TallyServer.get_start_config()
        '''
        if not self.is_participating(client_uid) or client_uid not in self.need_shares:
            return None

        assert self.state == 'starting_dcs' or self.state == 'starting_sks'
        config = {}

        cname = TallyServer.get_client_display_name(client_uid)

        if self.state == 'starting_dcs' and client_uid in self.dc_uids:
            config['sharekeepers'] = {}
            for sk_uid in self.sk_public_keys:
                config['sharekeepers'][sk_uid] = b64encode(self.sk_public_keys[sk_uid])
            config['counters'] = self.counters_config
            if self.traffic_model_config is not None:
                config['traffic_model'] = self.traffic_model_config
            config['noise'] = self.noise_config
            config['noise_weight'] = self.noise_weight_config
            config['dc_threshold'] = self.dc_threshold_config
            config['defer_time'] = self.clock_padding
            config['collect_period'] = self.period
            config['max_cell_events_per_circuit'] = self.max_cell_events_per_circuit
            config['circuit_sample_rate'] = self.circuit_sample_rate
            logging.info("sending start comand with {} counters ({} bins) and requesting {} shares to data collector {}"
                         .format(len(config['counters']),
                                 count_bins(config['counters']),
                                 len(config['sharekeepers']),
                                 cname))
            logging.debug("full data collector start config {}".format(config))

        elif self.state == 'starting_sks' and client_uid in self.sk_uids:
            config['shares'] = self.encrypted_shares[client_uid]
            config['counters'] = self.counters_config
            if self.traffic_model_config is not None:
                config['traffic_model'] = self.traffic_model_config
            config['noise'] = self.noise_config
            config['noise_weight'] = self.noise_weight_config
            config['dc_threshold'] = self.dc_threshold_config
            config['collect_period'] = self.period
            logging.info("sending start command with {} counters ({} bins) and {} shares to share keeper {}"
                         .format(len(config['counters']),
                                 count_bins(config['counters']),
                                 len(config['shares']),
                                 cname))
            logging.debug("full share keeper start config {}".format(config))

        return config

    def get_stop_config(self, client_uid):
        if not self.is_participating(client_uid) or client_uid not in self.need_counts:
            return None

        assert self.state == 'stopping'

        cname = TallyServer.get_client_display_name(client_uid)

        config = {'send_counters' : not self.is_error()}
        msg = "without" if self.is_error() else "with"
        logging.info("sending stop command to {} {} request for counters"
                     .format(cname, msg))
        return config

    def set_tally_server_status(self, status):
        '''
        status is a dictionary
        '''
        # make a deep copy, so we can delete unnecesary keys
        self.tally_server_status = deepcopy(status)

    def set_client_status(self, status):
        '''
        status is a dictionary of dictionaries, indexed by UID, and then by the
        attribute: name, fingerprint, ...
        '''
        self.client_status = deepcopy(status)

    def set_client_config(self, uid, config):
        '''
        config is a dictionary, indexed by the attributes: name, fingerprint, ...
        '''
        self.client_config[uid] = deepcopy(config)

    def get_client_types(self):
        '''
        returns a list of unique types of clients in self.client_status
        '''
        types = []
        if self.client_status is None:
            return types
        for uid in self.client_status:
            for k in self.client_status[uid].keys():
                if k == 'type' and not self.client_status[uid]['type'] in types:
                    types.append(self.client_status[uid]['type'])
        return types

    def get_client_context_by_type(self):
        '''
        returns a context for each client by UID, grouped by client type
        '''
        contexts = {}
        # we can't group by type without the type from the status
        if self.client_status is None:
            return contexts
        for type in self.get_client_types():
            for uid in self.client_status:
                if self.client_status[uid].get('type', 'NoType') == type:
                    contexts.setdefault(type, {}).setdefault(uid, {})['Status'] = self.client_status[uid]
                    # remove the (inner) types, because they're redundant now
                    del contexts[type][uid]['Status']['type']
                    # add the client config as well
                    if self.client_config is not None and uid in self.client_config:
                        contexts[type][uid]['Config'] = self.client_config[uid]
        return contexts

    def get_result_context(self, end_time):
        '''
        the context is written out with the tally results
        '''
        result_context = {}

        # log the times used for the round
        result_time = {}
        # Do we want to round these times?
        # (That is, use begin and end instead?)
        result_time['Start'] = self.starting_ts
        result_time['Stopping'] = self.stopping_ts
        result_time['End'] = end_time
        result_time['CollectStopping'] = self.stopping_ts - self.starting_ts
        result_time['CollectEnd'] = end_time - self.starting_ts
        result_time['StoppingDelay'] = end_time - self.stopping_ts
        # the collect, event, and checkin periods are in the tally server config
        result_time['ClockPadding'] = self.clock_padding
        result_context['Time'] = result_time

        # the bins are listed in each Tally, so we don't duplicate them here
        #result_count_context['CounterBins'] = self.counters_config

        # add the context for the clients that participated in the count
        # this includes all status information by default
        # clients are grouped by type, rather than listing them all by UID at
        # the top level of the context
        if self.client_status is not None:
            result_context.update(self.get_client_context_by_type())

        # now remove any context we are sure we don't want
        for dc in result_context.get('DataCollector', {}).values():
            # We don't need the paths from the configs
            if 'state' in dc.get('Config', {}):
                dc['Config']['state'] = "(state path)"
            if 'secret_handshake' in dc.get('Config', {}):
                dc['Config']['secret_handshake'] = "(secret_handshake path)"
            # or the counters
            if 'counters' in dc.get('Config', {}).get('Start',{}):
                dc['Config']['Start']['counters'] = "(counter bins, no counts)"
            if 'counters' in dc.get('Config', {}).get('Start',{}).get('noise',{}):
                dc['Config']['Start']['noise']['counters'] = "(counter sigmas, no counts)"
            # or the sk public keys
            if 'sharekeepers' in dc.get('Config', {}).get('Start',{}):
                for uid in dc['Config']['Start']['sharekeepers']:
                    dc['Config']['Start']['sharekeepers'][uid] = "(public key)"

        # We don't want the public key in the ShareKeepers' statuses
        for sk in result_context.get('ShareKeeper', {}).values():
            if 'key' in sk.get('Config', {}):
                sk['Config']['key'] = "(key path)"
            if 'state' in sk.get('Config', {}):
                sk['Config']['state'] = "(state path)"
            if 'secret_handshake' in sk.get('Config', {}):
                sk['Config']['secret_handshake'] = "(secret_handshake path)"
            if 'public_key' in sk.get('Status', {}):
                sk['Status']['public_key'] = "(public key)"
            # or the counters
            if 'counters' in sk.get('Config', {}).get('Start',{}):
                sk['Config']['Start']['counters'] = "(counter bins, no counts)"
            if 'counters' in sk.get('Config', {}).get('Start',{}).get('noise',{}):
                sk['Config']['Start']['noise']['counters'] = "(counter sigmas, no counts)"

        # add the status and config for the tally server itself
        result_context['TallyServer'] = {}
        if self.tally_server_status is not None:
            result_context['TallyServer']['Status'] = self.tally_server_status
        # even though the counter limits are hard-coded, include them anyway
        result_context['TallyServer']['Config'] = add_counter_limits_to_config(self.tally_server_config)

        # We don't need the paths from the configs
        if 'cert' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['cert'] = "(cert path)"
        if 'key' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['key'] = "(key path)"
        if 'state' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['state'] = "(state path)"
        if 'secret_handshake' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['secret_handshake'] = \
                "(secret_handshake path)"
        if 'allocation' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['allocation'] = \
                "(allocation path)"
        if 'results' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['results'] = \
                "(results path)"
        # And we don't need the bins, they're duplicated in 'Tally'
        if 'counters' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['counters'] = "(counter bins, no counts)"
        # but we want the noise, because it's not in Tally

        return result_context

    def get_updated_traffic_model(self, tallied_counts):
        '''
        Given the tallied counters in tallied_counts, compute an updated
        TrafficModel config by loading the initial traffic model and updating
        the states of the model based on the traffic model labels that were
        just counted during this round.

        Return None if a traffic model config was not provided as input this round,
        or if there was a problem with the tallied_counts that would prevent us from
        updating the model, or if there was an exception in the traffic model
        update function.

        Return the updated traffic model config on success. The new config can
        be used as input into the next collection round and can be used to
        instantiate another TrafficModel instance.
        '''
        if self.traffic_model_config is None: return None

        # create a TrafficModel object from the original input model config
        tmodel = TrafficModel(self.traffic_model_config)
        all_tmodel_labels = tmodel.get_all_counter_labels()

        # the traffic model class expects counts only, i.e, dict[label] = count
        tmodel_counts = {}
        for label in all_tmodel_labels:
            if label not in tallied_counts:
                logging.warning("tallied counters are missing traffic model label {}"
                                .format(label))
            elif 'bins' not in tallied_counts[label]:
                logging.warning("tallied counters are missing bins for traffic model label {}"
                                .format(label))
            elif len(tallied_counts[label]['bins']) < 1:
                logging.warning("tallied counters have too few bins for traffic model label {}"
                                .format(label))
            elif len(tallied_counts[label]['bins'][0]) < 3:
                logging.warning("tallied counters are missing bin count for traffic model label {}"
                                .format(label))
            else:
                # get the actual count (traffic model only uses 1 bin for each label)
                tmodel_counts[label] = tallied_counts[label]['bins'][0][2]

        # now make sure we got counts for all of the labels
        if len(tmodel_counts) == len(all_tmodel_labels):
            # update the original tmodel based on our new counts, and output. it's
            # OK if this fails, because the counts will be stored in the results
            # context and can be used to update the model after the round ends
            try:
                updated_tmodel_conf = tmodel.update_from_tallies(tmodel_counts)
                return updated_tmodel_conf
            except:
                logging.warning("there was a non-fatal exception in the traffic model update function")
                log_error()
                return None
        else:
            # some problem with counter labels
            logging.warning("the traffic model and tallied counter labels are inconsistent")
            return None

    def write_json_file(self, json_object, path_prefix, filename_prefix, begin, end):
        filepath = os.path.join(path_prefix,
                                "{}.{}-{}.json"
                                .format(filename_prefix, begin, end))
        with open(filepath, 'w') as fout:
            json.dump(json_object, fout, sort_keys=True, indent=4)

        return filepath

    def write_results(self, path_prefix, end_time):
        '''
        Write collections results to a file in path_prefix, including end_time
        in the context.
        '''
        # this should already have been done, but let's make sure
        path_prefix = normalise_path(path_prefix)

        if not self.is_stopped():
            logging.warning("trying to write results before collection phase is stopped")
            return

        # keep going, we want the context for debugging
        tally_was_successful = False
        if len(self.final_counts) <= 0:
            logging.warning("no tally results to write!")
        else:
            tallied_counter = SecureCounters(self.counters_config,
                                             self.modulus,
                                             require_generate_noise=False)
            tally_was_successful = tallied_counter.tally_counters(
                self.final_counts.values())

        begin = int(round(self.starting_ts))
        end = int(round(self.stopping_ts))

        tallied_counts = {}
        # keep going, we want the context for debugging
        if not tally_was_successful:
            logging.warning("problem tallying counters, did all counters and bins match!?")
        else:
            tallied_counts = tallied_counter.detach_counts()

            # For backwards compatibility, write out a "tallies" file
            # This file only has the counts
            self.write_json_file(tallied_counts, path_prefix,
                                 "privcount.tallies", begin, end)

        #logging.info("tally was successful, counts for phase from %d to %d were written to file '%s'", begin, end, filepath)

        # Write out an "outcome" file that adds context to the counts
        # This makes it easier to interpret results later on
        result_info = {}

        if tally_was_successful:
            # add the existing list of counts as its own item
            result_info['Tally'] = tallied_counts

            if self.traffic_model_config is not None:
                # compute the updated traffic model and store in results context
                result_info['UpdatedTrafficModel'] = self.get_updated_traffic_model(tallied_counts)

                # also write out a copy of the new model
                self.write_json_file(result_info['UpdatedTrafficModel'],
                                     path_prefix, "privcount.traffic.model", begin, end)

        # add the context of the outcome as another item
        result_info['Context'] = self.get_result_context(end_time)

        filepath = self.write_json_file(result_info, path_prefix,
                             "privcount.outcome", begin, end)

        logging.info("tally {}, outcome of phase of {} was written to file '{}'"
                     .format(
                     "was successful" if tally_was_successful else "failed",
                     format_interval_time_between(begin, 'from', end),
                     filepath))
        self.final_counts = {}

    def log_status(self):
        message = "collection phase is in '{}' state".format(self.state)

        if self.state == 'starting_dcs':
            message += ", waiting to receive shares from {} DCs: {}".format(len(self.need_shares), ','.join([ TallyServer.get_client_display_name(uid) for uid in self.need_shares]))
        elif self.state == 'starting_sks':
            message += ", waiting to send shares to {} SKs: {}".format(len(self.need_shares), ','.join([ TallyServer.get_client_display_name(uid) for uid in self.need_shares]))
        elif self.state == 'started':
            message += ", running for {}".format(format_elapsed_time_since(self.starting_ts, 'since'))
        elif self.state == 'stopping':
            message += ", trying to stop for {}".format(format_elapsed_time_since(self.stopping_ts, 'since'))
            message += ", waiting to receive counts from {} DCs/SKs: {}".format(len(self.need_counts), ','.join([ TallyServer.get_client_display_name(uid) for uid in self.need_counts]))

        logging.info(message)
