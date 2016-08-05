'''
Created on Dec 12, 2015

@author: rob
'''
import os
import json
import logging
import cPickle as pickle

from time import time
from base64 import b64encode

from twisted.internet import reactor, task, ssl
from twisted.internet.protocol import ServerFactory

from protocol import PrivCountServerProtocol
from util import log_error, SecureCounters, TrafficModel, generate_keypair, generate_cert

import yaml

# for warning about logging function and format # pylint: disable=W1202
# for calling methods on reactor # pylint: disable=E1101

def log_tally_server_status(status):
    minutes = int((time() - status['time'])/ 60.0)
    logging.info("--server status: PrivCount is {} for {} minutes (since {})".format(status['state'], minutes, status['time']))
    t, r = status['dcs_total'], status['dcs_required']
    a, i = status['dcs_active'], status['dcs_idle']
    logging.info("--server status: DataCollectors: have {}, need {}, {}/{} active, {}/{} idle".format(t, r, a, t, i, t))
    t, r = status['sks_total'], status['sks_required']
    a, i = status['sks_active'], status['sks_idle']
    logging.info("--server status: ShareKeepers: have {}, need {}, {}/{} active, {}/{} idle".format(t, r, a, t, i, t))

class TallyServer(ServerFactory):
    '''
    receive blinded counts from the DCs
    receive key shares from the SKs
    sum shares and counts at end of epoch
    publish the final results to a file
    '''

    def __init__(self, config_filepath):
        self.config_filepath = config_filepath
        self.config = None
        self.clients = {}
        self.collection_phase = None
        self.idle_time = time()
        self.num_completed_collection_phases = 0

    def buildProtocol(self, addr):
        return PrivCountServerProtocol(self)

    def startFactory(self):
        # TODO
        return
        # load any state we may have from a previous run
        state_filepath = self.config['state']
        if os.path.exists(state_filepath):
            with open(state_filepath, 'r') as fin:
                state = pickle.load(fin)
                self.clients = state['clients']
                self.collection_phase = state['collection_phase']
                self.idle_time = state['idle_time']

    def stopFactory(self):
        # TODO
        return
        state_filepath = self.config['state']
        if self.collection_phase is not None or len(self.clients) > 0:
            # export everything that would be needed to survive an app restart
            state = {'clients': self.clients, 'collection_phase': self.collection_phase, 'idle_time': self.idle_time}
            with open(state_filepath, 'w') as fout:
                pickle.dump(state, fout)

    def run(self):
        # load initial config
        self.refresh_config()
        if self.config is None:
            return

        # refresh and check status every minute
        task.LoopingCall(self.refresh_loop).start(60, now=False)

        # setup server for receiving blinded counts from the DC nodes and key shares from the SK nodes
        listen_port = self.config['listen_port']
        key_path = self.config['key']
        cert_path = self.config['cert']
        ssl_context = ssl.DefaultOpenSSLContextFactory(key_path, cert_path)

        logging.info("Tally Server listening on port {}".format(listen_port))
        reactor.listenSSL(listen_port, self, ssl_context)
        reactor.run()

    def refresh_loop(self):
        # make sure we have the latest config and counters
        self.refresh_config()

        # check if any clients have not checked in recently
        self.clear_dead_clients()

        # check if we should start the next collection phase
        if self.collection_phase is None:
            num_phases = self.num_completed_collection_phases
            if num_phases == 0 or self.config['continue']:
                dcs, sks = self.get_idle_dcs(), self.get_idle_sks()
                if len(dcs) >= self.config['dc_threshold'] and len(sks) >= self.config['sk_threshold']:
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
        # re-read config and process any changes
        try:
            logging.debug("reading config file from '%s'", self.config_filepath)

            # read in the config from the given path
            with open(self.config_filepath, 'r') as fin:
                conf = yaml.load(fin)
            ts_conf = conf['tally_server']

            if 'counters' in ts_conf:
                expanded_path = os.path.expanduser(ts_conf['counters'])
                ts_conf['counters'] = os.path.abspath(expanded_path)
                assert os.path.exists(os.path.dirname(ts_conf['counters']))
                with open(ts_conf['counters'], 'r') as fin:
                    counters_conf = yaml.load(fin)
                ts_conf['counters'] = counters_conf['counters']
            else:
                ts_conf['counters'] = conf['counters']

            # if key path is not specified, look at default path, or generate a new key
            if 'key' in ts_conf and 'cert' in ts_conf:
                expanded_path = os.path.expanduser(ts_conf['key'])
                ts_conf['key'] = os.path.abspath(expanded_path)
                assert os.path.exists(ts_conf['key'])

                expanded_path = os.path.expanduser(ts_conf['cert'])
                ts_conf['cert'] = os.path.abspath(expanded_path)
                assert os.path.exists(ts_conf['cert'])
            else:
                ts_conf['key'] = 'privcount.rsa_key.pem'
                ts_conf['cert'] = 'privcount.rsa_key.cert'
                if not os.path.exists(ts_conf['key']) or not os.path.exists(ts_conf['cert']):
                    generate_keypair(ts_conf['key'])
                    generate_cert(ts_conf['key'], ts_conf['cert'])

            if 'results' in ts_conf:
                expanded_path = os.path.expanduser(ts_conf['results'])
                ts_conf['results'] = os.path.abspath(expanded_path)
                assert os.path.exists(os.path.dirname(ts_conf['results']))

            if 'traffic_model' in ts_conf:
                expanded_path = os.path.expanduser(ts_conf['traffic_model'])
                ts_conf['traffic_model'] = os.path.abspath(expanded_path)
                assert os.path.exists(os.path.dirname(ts_conf['traffic_model']))

                inf = open(ts_conf['traffic_model'], 'r')
                model = json.load(inf)
                inf.close()
                assert 'states' in model
                assert 'start_probability' in model
                assert 'transition_probability' in model
                assert 'emission_probability' in model

            expanded_path = os.path.expanduser(ts_conf['state'])
            ts_conf['state'] = os.path.abspath(expanded_path)
            assert os.path.exists(os.path.dirname(ts_conf['state']))

            assert ts_conf['listen_port'] > 0
            assert ts_conf['sk_threshold'] > 0
            assert ts_conf['dc_threshold'] > 0
            assert ts_conf['checkin_period'] > 0
            assert ts_conf['collect_period'] > 0
            assert ts_conf['continue'] == True or ts_conf['continue'] == False
            assert ts_conf['q'] > 0

            for key in ts_conf['counters']:
                if 'Histogram' in key:
                    assert 'bins' in ts_conf['counters'][key] and ts_conf['counters'][key]['bins'] is not None

            if self.config == None:
                self.config = ts_conf
                logging.info("using config = %s", str(self.config))
            else:
                changed = False
                for k in ts_conf:
                    if k not in self.config or ts_conf[k] != self.config[k]:
                        logging.info("updated config for key {} from {} to {}".format(k, self.config[k], ts_conf[k]))
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

    def clear_dead_clients(self):
        now = time()

        for uid in self.clients.keys():
            c_status = self.clients[uid]
            time_since_checkin = now - c_status['alive']

            if time_since_checkin > 2 * self.get_checkin_period():
                logging.warning("last checkin was {} minutes ago for client {}".format(time_since_checkin/60.0, c_status))

            if time_since_checkin > 6 * self.get_checkin_period():
                logging.warning("marking dead client {}".format(c_status))
                c_status['state'] = 'dead'

                if self.collection_phase is not None and self.collection_phase.is_participating(uid):
                    self.collection_phase.lost_client(uid)

                self.clients.pop(uid, None)

    def _get_matching_clients(self, c_type, c_state):
        matching_clients = []
        for uid in self.clients:
            if self.clients[uid]['type'] == c_type and self.clients[uid]['state'] == c_state:
                matching_clients.append(uid)
        return matching_clients

    def get_idle_dcs(self):
        return self._get_matching_clients('DataCollector', 'idle')

    def get_active_dcs(self):
        return self._get_matching_clients('DataCollector', 'active')

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
            'sks_idle' : sk_idle,
            'sks_active' : sk_active,
            'sks_total' : sk_idle+sk_active,
            'sks_required' : self.config['sk_threshold']
        }

        return status

    def set_client_status(self, uid, status): # called by protocol
        oldstate = self.clients[uid]['state'] if uid in self.clients else status['state']
        oldtime = self.clients[uid]['time'] if uid in self.clients else status['alive']

        if uid not in self.clients:
            logging.info("new {} {} joined and is {}".format(status['type'], uid, status['state']))

        self.clients[uid] = status
        if oldstate == self.clients[uid]['state']:
            self.clients[uid]['time'] = oldtime
        else:
            self.clients[uid]['time'] = status['alive']

        minutes = int((time() - status['time'])/ 60.0)
        logging.info("----client status: {} {} is alive and {} for {} minutes (since {})".format(status['type'], uid, status['state'], minutes, status['time']))

    def get_clock_padding(self, client_uids):
        max_delay = max([self.clients[uid]['rtt']+self.clients[uid]['clock_skew'] for uid in client_uids])
        return max_delay + self.get_checkin_period()

    def start_new_collection_phase(self, dc_uids, sk_uids):
        assert self.collection_phase is None

        clock_padding = self.get_clock_padding(dc_uids + sk_uids)

        sk_public_keys = {}
        for uid in sk_uids:
            sk_public_keys[uid] = self.clients[uid]['public_key']

        counters = self.config['counters']
        model = None
        if 'traffic_model' in self.config:
            # load the model json file
            inf = open(self.config['traffic_model'], 'r')
            model = json.load(inf)
            inf.close()

            # add all of the counters needed to count the model states
            for label in TrafficModel(model['states'], model['start_probability'], model['transition_probability'], model['emission_probability']).get_counter_labels():
                counters[label] = {'bins': [[0.0, float("inf")]]}

        self.collection_phase = CollectionPhase(self.config['collect_period'], counters, sk_uids, sk_public_keys, dc_uids, self.config['q'], clock_padding, model)
        self.collection_phase.start()

    def stop_collection_phase(self):
        assert self.collection_phase is not None
        self.collection_phase.stop()
        if self.collection_phase.is_stopped():
            self.num_completed_collection_phases += 1
            dir_path = './' if 'results' not in self.config else self.config['results']
            self.collection_phase.write_results(dir_path)
            self.collection_phase = None
            self.idle_time = time()

    def get_start_config(self, client_uid): # called by protocol
        # return None to indicate we shouldnt start the client yet
        if self.collection_phase is not None:
            return self.collection_phase.get_start_config(client_uid)
        else:
            return None

    def set_start_result(self, client_uid, result_data): # called by protocol
        if self.collection_phase is not None:
            self.collection_phase.store_data(client_uid, result_data)

    def get_stop_config(self, client_uid): # called by protocol
        # returns None to indicate we shouldnt stop the client yet
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

    def __init__(self, period, counters_config, sk_uids, sk_public_keys, dc_uids, param_q, clock_padding, traffic_model):
        # store configs
        self.period = period
        self.counters_config = counters_config
        self.sk_uids = sk_uids
        self.sk_public_keys = sk_public_keys
        self.dc_uids = dc_uids
        self.param_q = param_q
        self.clock_padding = clock_padding
        self.traffic_model = traffic_model

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
        # this is called when client_uid isnt responding
        # we could mark error_flag as true and abort, or keep counting anyway
        # and hope we can recover from the error by adding the local state
        # files later... TODO
        pass

    def store_data(self, client_uid, data):
        if data == None:
            logging.warning("received error response from {} while in state {}".format(client_uid, self.state))
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
                logging.info("received {} shares from data collector {}".format(len(shares), client_uid))

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
            logging.info("share keeper {} started and received its shares".format(client_uid))
            self.need_shares.remove(client_uid)
            if len(self.need_shares) == 0:
                self._change_state('started')

        elif self.state == 'stopping':
            if client_uid in self.need_counts:
                # the client got our stop command
                counts = data
                logging.info("received {} counts from stopped client {}".format(len(counts), client_uid))

                if not self.is_error() and len(counts) == 0:
                    logging.warning("received empty counts from {}, final results will not be available".format(client_uid))
                    self.error_flag = True
                if not self.is_error():
                    # add up the tallies from the client
                    self.final_counts[client_uid] = data
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

    def get_start_ts(self):
        return self.starting_ts

    def get_start_config(self, client_uid):
        if not self.is_participating(client_uid) or client_uid not in self.need_shares:
            return None

        assert self.state == 'starting_dcs' or self.state == 'starting_sks'
        config = {'q':self.param_q}

        if self.state == 'starting_dcs' and client_uid in self.dc_uids:
            config['sharekeepers'] = {}
            for sk_uid in self.sk_public_keys:
                config['sharekeepers'][sk_uid] = b64encode(self.sk_public_keys[sk_uid])
            config['counters'] = self.counters_config
            config['defer_time'] = self.clock_padding
            if self.traffic_model is not None:
                config['traffic_model'] = self.traffic_model
            logging.info("sending start comand with {} counters and requesting {} shares to data collector {}".format(len(config['counters']), len(config['sharekeepers']), client_uid))

        elif self.state == 'starting_sks' and client_uid in self.sk_uids:
            config['shares'] = self.encrypted_shares[client_uid]
            config['counters'] = self.counters_config
            logging.info("sending start command with {} counters and {} shares to share keeper {}".format(len(config['counters']), len(config['shares']), client_uid))

        return config

    def get_stop_config(self, client_uid):
        if not self.is_participating(client_uid) or client_uid not in self.need_counts:
            return None

        assert self.state == 'stopping'
        config = {'send_counters' : not self.is_error()}
        msg = "without" if self.is_error() else "with"
        logging.info("sending stop command to {} {} request for counters".format(client_uid, msg))
        return config

    def write_results(self, path_prefix):
        if not self.is_stopped():
            logging.warning("trying to write results before collection phase is stopped")
            return
        if len(self.final_counts) <= 0:
            logging.warning("no tally results to write!")
            return

        tallied_counter = SecureCounters(self.counters_config, self.param_q)
        tally_was_successful = tallied_counter.tally_counters(self.final_counts.values())

        if not tally_was_successful:
            logging.warning("problem tallying counters, did all counters and bins match!?")
            return

        tallied_counts = tallied_counter.detach_counts()

        begin, end = int(round(self.starting_ts)), int(round(self.stopping_ts))
        filepath = "{}/privcount.tallies.{}-{}.json".format(path_prefix, begin, end)
        with open(filepath, 'a') as fout:
            json.dump(tallied_counts, fout, sort_keys=True, indent=4)

        logging.info("tally was successful, results for phase from %d to %d were written to file '%s'", begin, end, filepath)
        self.final_counts = {}

    def log_status(self):
        message = "collection phase is in '{}' state".format(self.state)

        if self.state == 'starting_dcs':
            message += ", waiting to receive shares from {} DCs: {}".format(len(self.need_shares), ','.join([ uid for uid in self.need_shares]))
        elif self.state == 'starting_sks':
            message += ", waiting to send shares to {} SKs: {}".format(len(self.need_shares), ','.join([ uid for uid in self.need_shares]))
        elif self.state == 'started':
            minutes = (time() - self.starting_ts) / 60.0
            message += ", running for {} minutes (since {})".format(minutes, self.starting_ts)
        elif self.state == 'stopping':
            minutes = (time() - self.stopping_ts) / 60.0
            message += ", trying to stop for {} minutes (since {})".format(minutes, self.stopping_ts)
            message += ", waiting to receive counts from {} DCs/SKs: {}".format(len(self.need_counts), ','.join([ uid for uid in self.need_counts]))

        logging.info(message)
