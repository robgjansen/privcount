import os
import logging
import math
import string
import cPickle as pickle

from time import time
from copy import deepcopy
from base64 import b64decode

from protocol import PrivCountClientProtocol, TorControlClientProtocol
from tally_server import log_tally_server_status
from util import SecureCounters, TrafficModel, log_error, get_public_digest_string, load_public_key_string, encrypt, format_delay_time_wait, format_last_event_time_since, normalise_path, counter_modulus, add_counter_limits_to_config

import yaml

from twisted.internet import task, reactor, ssl
from twisted.internet.protocol import ReconnectingClientFactory

# using reactor: pylint: disable=E1101
# method docstring missing: pylint: disable=C0111
# line too long: pylint: disable=C0301

class DataCollector(ReconnectingClientFactory):
    '''
    receive key share data from the DC message receiver
    keep the shares during collection epoch
    send the shares to the TS at end of epoch
    '''

    def __init__(self, config_filepath):
        self.config_filepath = normalise_path(config_filepath)
        self.config = None
        self.aggregator = None
        self.aggregator_defer_id = None
        self.context = {}

    def buildProtocol(self, addr):
        return PrivCountClientProtocol(self)

    def startFactory(self):
        # TODO
        return
        # load any state we may have from a previous run
        state_filepath = normalise_path(self.config['state'])
        if os.path.exists(state_filepath):
            with open(state_filepath, 'r') as fin:
                state = pickle.load(fin)
                self.aggregator = state['aggregator']
                self.aggregator_defer_id = state['aggregator_defer_id']

    def stopFactory(self):
        # TODO
        return
        state_filepath = normalise_path(self.config['state'])
        if self.aggregator is not None:
            # export everything that would be needed to survive an app restart
            state = {'aggregator': self.aggregator, 'aggregator_defer_id': self.aggregator_defer_id}
            with open(state_filepath, 'w') as fout:
                pickle.dump(state, fout)

    def run(self):
        # load iniital config
        self.refresh_config()
        if self.config is None:
            logging.critical("cannot start due to error in config file")
            return

        # connect to the tally server, register, and wait for commands
        self.do_checkin()
        reactor.run()

    def get_status(self): # called by protocol
        status = {'type':'DataCollector', 'name':self.config['name'],
                  'state': 'active' if self.aggregator is not None else 'idle'}
        # store the latest context, so we have it even when the aggregator goes away
        if self.aggregator is not None:
            self.context.update(self.aggregator.get_context())
        # and include the latest context values in the status
        status.update(self.context)
        return status

    def set_server_status(self, status): # called by protocol
        log_tally_server_status(status)

    def do_checkin(self): # called by protocol
        self.refresh_config()
        # turn on reconnecting mode and reset backoff
        self.resetDelay()
        ts_ip = self.config['tally_server_info']['ip']
        ts_port = self.config['tally_server_info']['port']
        logging.info("checking in with TallyServer at {}:{}".format(ts_ip, ts_port))
        reactor.connectSSL(ts_ip, ts_port, self, ssl.ClientContextFactory()) # pylint: disable=E1101

    def do_start(self, config): # called by protocol
        '''
        start the node running
        return None if failure, otherwise json will encode result
        '''
        if 'sharekeepers' not in config or 'counters' not in config:
            return None
        # if we are still running from a previous incarnation, we need to stop first
        if self.aggregator is not None:
            return None

        ts_counters, dc_counters = deepcopy(config['counters']), deepcopy(self.config['counters'])

        # we keep our local config for which keys we are counting and the sigmas for noise
        # we allow the tally server to update only the bin widths for each counter
        for key in ts_counters:
            if key in dc_counters and 'bins' in ts_counters[key]:
                dc_counters[key]['bins'] = deepcopy(ts_counters[key]['bins'])

        # we cant count keys for which we don't have configured bins
        for key in dc_counters:
            if 'bins' not in dc_counters[key]:
                logging.warning("skipping counter '{}' because we do not have any bins configured".format(key))
                dc_counters.pop(key, None)

        # we require that only the configured share keepers be used in the collection phase
        # because we must be able to encrypt messages to them
        expected_sk_digests = set()
        for digest in self.config['share_keepers']:
            expected_sk_digests.add(digest)

        # verify that we have the public cert for each share keeper that the TS wants to use
        digest_error = False
        for sk_uid in config['sharekeepers']:
            pub_key_str = b64decode(config['sharekeepers'][sk_uid])
            requested_sk_digest = get_public_digest_string(pub_key_str, is_private_key=False)

            if requested_sk_digest not in expected_sk_digests:
                logging.info('we received an unexpected key for share keeper {}'.format(sk_uid))
                digest_error = True

            expected_sk_digests.remove(requested_sk_digest)

        if digest_error or len(expected_sk_digests) != 0:
            logging.info('refusing to start collecting without required share keepers')
            return None

        # validate the traffic model we got from the tally server
        traffic_model_valid = False
        if 'traffic_model' in config:
            traffic_model_valid = True
            for k in ['states', 'emission_probability', 'transition_probability', 'start_probability']:
                if k not in config['traffic_model']:
                    traffic_model_valid = False
        # load the traffic model object that we will use during aggregation
        tmodel = None
        if traffic_model_valid:
            # build the model
            states = config['traffic_model']['states']
            start_p = config['traffic_model']['start_probability']
            trans_p = config['traffic_model']['transition_probability']
            emit_p = config['traffic_model']['emission_probability']
            tmodel = TrafficModel(states, start_p, trans_p, emit_p)

            # now add in the keys needed for counting
            # NOTE the tally server sent these, but we ignored them above
            for label in tmodel.get_counter_labels():
                # XXX TODO FIXME these sigmas should not be 0
                dc_counters[label] = {'bins': [[0.0, float("inf")]], 'sigma': 0.0}

        self.aggregator = Aggregator(dc_counters, tmodel, config['sharekeepers'], self.config['noise_weight'], counter_modulus(), self.config['event_source'])

        defer_time = config['defer_time'] if 'defer_time' in config else 0.0
        logging.info("got start command from tally server, starting aggregator in {}".format(format_delay_time_wait(defer_time, 'at')))

        # sync the time that we start listening for Tor events
        self.aggregator_defer_id = reactor.callLater(defer_time, self._start_aggregator_deferred)

        # return the generated shares now
        shares = self.aggregator.get_shares()
        # this is a dict {sk_uid : sk_msg} for each sk
        for sk_uid in shares:
            # encrypt shares[sk_uid] for that sk
            pub_key_str = b64decode(config['sharekeepers'][sk_uid])
            sk_pub_key = load_public_key_string(pub_key_str)
            encrypted_secret = encrypt(sk_pub_key, shares[sk_uid]['secret'])
            shares[sk_uid]['secret'] = encrypted_secret

        logging.info("successfully started and generated {} blinding shares for {} counters".format(len(shares), len(dc_counters)))
        return shares

    def _start_aggregator_deferred(self):
        self.aggregator_defer_id = None
        self.aggregator.start()

    def do_stop(self, config): # called by protocol
        '''
        stop the node from running
        return None if failure, otherwise json will encode result
        '''
        logging.info("got command to stop collection phase")
        if 'send_counters' not in config:
            return None

        wants_counters = 'send_counters' in config and config['send_counters'] is True
        logging.info("tally server {} final counts".format("wants" if wants_counters else "does not want"))

        response = {}
        if self.aggregator_defer_id is not None:
            self.aggregator_defer_id.cancel()
            self.aggregator_defer_id = None
            assert self.aggregator is None

        elif self.aggregator is not None:
            counts = self.aggregator.stop()
            del self.aggregator
            self.aggregator = None
            if wants_counters:
                logging.info("sending counts from {} counters".format(len(counts)))
                response['Counts'] = counts

        logging.info("collection phase was stopped")
        # even though the counter limits are hard-coded, include them anyway
        response['Config'] = add_counter_limits_to_config(self.config)
        return response

    def refresh_config(self):
        '''
        re-read config and process any changes
        '''
        try:
            logging.debug("reading config file from '%s'", self.config_filepath)

            # read in the config from the given path
            with open(self.config_filepath, 'r') as fin:
                conf = yaml.load(fin)
            dc_conf = conf['data_collector']

            if 'counters' in dc_conf:
                dc_conf['counters'] = normalise_path(dc_conf['counters'])
                assert os.path.exists(os.path.dirname(dc_conf['counters']))
                with open(dc_conf['counters'], 'r') as fin:
                    counters_conf = yaml.load(fin)
                dc_conf['counters'] = counters_conf['counters']
            else:
                dc_conf['counters'] = conf['counters']

            dc_conf['state'] = normalise_path(dc_conf['state'])
            assert os.path.exists(os.path.dirname(dc_conf['state']))

            assert dc_conf['name'] != ''
            assert dc_conf['noise_weight'] >= 0
            assert dc_conf['tally_server_info']['ip'] is not None
            assert dc_conf['tally_server_info']['port'] > 0

            assert dc_conf['event_source'] is not None
            assert dc_conf['event_source'] > 0

            for key in dc_conf['counters']:
                assert dc_conf['counters'][key]['sigma'] >= 0.0

            assert 'share_keepers' in dc_conf

            if self.config == None:
                self.config = dc_conf
                logging.info("using config = %s", str(self.config))
            else:
                changed = False
                for k in dc_conf:
                    if k not in self.config or dc_conf[k] != self.config[k]:
                        logging.info("updated config for key {} from {} to {}".format(k, self.config[k], dc_conf[k]))
                        self.config[k] = dc_conf[k]
                        changed = True
                if not changed:
                    logging.debug('no config changes found')

        except AssertionError:
            logging.warning("problem reading config file: invalid data")
            log_error()
        except KeyError:
            logging.warning("problem reading config file: missing required keys")
            log_error()

class Aggregator(ReconnectingClientFactory):
    '''
    receive data from Tor control port
    parse the contents for valid events and stats
    aggregate stats during collection epoch
    add noise to aggregated stats at end of epoch
    send results for tallying
    '''

    def __init__(self, counters, traffic_model, sk_uids, noise_weight, param_q, tor_control_port):
        self.traffic_model = traffic_model

        self.secure_counters = SecureCounters(counters, param_q)
        self.secure_counters.generate(sk_uids, noise_weight)

        self.connector = None
        self.protocol = None
        self.rotator = None
        self.tor_control_port = tor_control_port

        self.last_event_time = 0.0
        self.num_rotations = 0L
        self.circ_info = {}
        self.strm_bytes = {}
        self.cli_ips_rotated = time()
        self.cli_ips_current = {}
        self.cli_ips_previous = {}

        self.nickname = None
        self.orport = None
        self.dirport = None
        self.version = None
        self.address = None
        self.fingerprint = None

    def buildProtocol(self, addr):
        self.protocol = TorControlClientProtocol(self)
        return self.protocol

    def startFactory(self):
        # TODO
        return

    def stopFactory(self):
        # TODO
        return

    def start(self):
        self.connector = reactor.connectTCP("127.0.0.1", self.tor_control_port, self)
        self.rotator = task.LoopingCall(self._do_rotate).start(600, now=False)
        self.cli_ips_rotated = time()

    def stop(self):
        # dont try to reconnect
        self.stopTrying()

        # stop reading from Tor control port
        if self.protocol is not None:
            self.protocol.quit()
        if self.rotator is not None:
            self.rotator.cancel()
        if self.connector is not None:
            self.connector.disconnect()

        # return the final counts and make sure we cant be restarted
        counts = self.secure_counters.detach_counts()
        del self.secure_counters
        self.secure_counters = None
        return counts

    def get_shares(self):
        return self.secure_counters.detach_blinding_shares()

    def set_nickname(self, nickname):
        nickname = nickname.strip()

        # Do some basic validation of the nickname
        if len(nickname) < 1 or len(nickname) > 19:
            logging.warning("Bad nickname length %d: %s", len(nickname), nickname)
            return False
        if not all(c in (string.ascii_letters + string.digits) for c in nickname):
            logging.warning("Bad nickname characters: %s", nickname)
            return False

        # Are we replacing an existing nickname?
        if self.nickname is not None:
            if self.nickname != nickname:
                logging.warning("Replacing nickname %s with %s", self.nickname, nickname)
            else:
                logging.debug("Duplicate nickname received %s", nickname)

        self.nickname = nickname

        return True

    def get_nickname(self):
        return self.nickname

    def set_orport(self, orport):
        orport = orport.strip()

        # Do some basic validation of the orport
        if len(orport) < 1 or len(orport) > 5:
            logging.warning("Bad orport length %d: %s", len(orport), orport)
            return False
        if not all(c in string.digits for c in orport):
            logging.warning("Bad orport characters: %s", orport)
            return False
        orport_n = int(orport)
        if orport_n < 1 or orport_n > 65535:
            logging.warning("Bad orport: out of range: %s", orport)
            return False

        # Are we replacing an existing nickname?
        if self.orport is not None:
            if self.orport != orport:
                logging.warning("Replacing orport %s with %s", self.orport, orport)
            else:
                logging.debug("Duplicate orport received %s", orport)

        self.orport = orport

        return True

    def get_orport(self):
        return self.orport

    def set_dirport(self, dirport):
        dirport = dirport.strip()

        # Do some basic validation of the dirport
        if len(dirport) < 1 or len(dirport) > 5:
            logging.warning("Bad dirport length %d: %s", len(dirport), dirport)
            return False
        if not all(c in string.digits for c in dirport):
            logging.warning("Bad dirport characters: %s", dirport)
            return False
        dirport_n = int(dirport)
        if dirport_n < 1 or dirport_n > 65535:
            logging.warning("Bad dirport: out of range: %s", dirport)
            return False

        # Are we replacing an existing nickname?
        if self.dirport is not None:
            if self.dirport != dirport:
                logging.warning("Replacing dirport %s with %s", self.dirport, dirport)
            else:
                logging.debug("Duplicate dirport received %s", dirport)

        self.dirport = dirport

        return True

    def get_dirport(self):
        return self.dirport

    def set_version(self, version):
        version = version.strip()

        # Do some basic validation of the version
        # This is hard, because versions can be almost anything
        if not len(version) > 0:
            logging.warning("Bad version length %d: %s", len(version), version)
            return False
        # This means unicode printables, there's no ASCII equivalent
        if not all(c in string.printable for c in version):
            logging.warning("Bad version characters: %s", version)
            return False

        # Are we replacing an existing version?
        if self.version is not None:
            if self.version != version:
                logging.warning("Replacing version %s with %s", self.version, version)
            else:
                logging.debug("Duplicate version received %s", version)

        self.version = version

        return True

    def get_version(self):
        return self.version

    def set_address(self, address):
        address = address.strip()

        # Do some basic validation of the address
        # Relays must all have IPv4 addresses, so just checking for IPv4 is ok
        if len(address) < 7 or len(address) > 15:
            logging.warning("Bad address length %d: %s", len(address), address)
            return False
        if not all(c in (string.digits + '.') for c in address):
            logging.warning("Bad address characters: %s", address)
            return False
        # We could check each component is between 0 and 255, but that's overkill

        # Are we replacing an existing address?
        if self.address is not None:
            if self.address != address:
                logging.warning("Replacing address %s with %s", self.address, address)
            else:
                logging.debug("Duplicate address received %s", address)

        self.address = address

        return True

    def get_address(self):
        return self.address

    def set_fingerprint(self, fingerprint):
        fingerprint = fingerprint.strip()

        # Do some basic validation of the fingerprint
        if not len(fingerprint) == 40:
            logging.warning("Bad fingerprint length %d: %s", len(fingerprint), fingerprint)
            return False
        if not all(c in string.hexdigits for c in fingerprint):
            logging.warning("Bad fingerprint characters: %s", fingerprint)
            return False

        # Are we replacing an existing fingerprint?
        if self.fingerprint is not None:
            if self.fingerprint != fingerprint:
                logging.warning("Replacing fingerprint %s with %s", self.fingerprint, fingerprint)
            else:
                logging.debug("Duplicate fingerprint received %s", fingerprint)

        self.fingerprint = fingerprint

        return True

    def get_fingerprint(self):
        return self.fingerprint

    def get_context(self):
        '''
        return a dictionary containing each available context item
        '''
        context = {}
        if self.get_nickname() is not None:
            context['nickname'] = self.get_nickname()
        if self.get_orport() is not None:
            context['orport'] = self.get_orport()
        if self.get_dirport() is not None:
            context['dirport'] = self.get_dirport()
        if self.get_version() is not None:
            context['version'] = self.get_version()
        if self.get_address() is not None:
            context['address'] = self.get_address()
        if self.get_fingerprint() is not None:
            context['fingerprint'] = self.get_fingerprint()
        if self.last_event_time != 0.0:
            context['last_event_time'] = self.last_event_time
        return context

    def handle_event(self, event):
        if not self.secure_counters:
            return False

        event_code, line_remaining = [v.strip() for v in event.split(' ', 1)]
        self.last_event_time = time()

        # hand valid events off to the aggregator
        if event_code == 's':
            # 's', ChanID, CircID, StreamID, ExitPort, ReadBW, WriteBW, TimeStart, TimeEnd, isDNS, isDir
            items = [v.strip() for v in line_remaining.split(' ', 10)]
            if len(items) == 10:
                self._handle_stream_event(items[0:10])

        elif event_code == 'c':
            # 'c', ChanID, CircID, nCellsIn, nCellsOut, ReadBWDNS, WriteBWDNS, ReadBWExit, WriteBWExit, TimeStart, TimeEnd, PrevIP, prevIsClient, prevIsRelay, NextIP, nextIsClient, nextIsRelay
            items = [v.strip() for v in line_remaining.split(' ', 16)]
            if len(items) == 16:
                self._handle_circuit_event(items[0:16])

        elif event_code == 't':
            # 't', ChanID, TimeStart, TimeEnd, IP, isClient, isRelay
            items = [v.strip() for v in line_remaining.split(' ', 6)]
            if len(items) == 6:
                self._handle_connection_event(items[0:6])

        elif event_code == 'b':
            # 'b', ChanID, CircID, StreamID, BW, Direction, Time
            items = [v.strip() for v in line_remaining.split(' ', 6)]
            if len(items) == 6:
                self._handle_bytes_event(items[0:6])

        return True

    def _handle_bytes_event(self, items):
        if self.traffic_model == None:
            return

        chanid, circid, strmid = [int(v) for v in items[0:3]]
        direction = items[3]
        bw_bytes = int(items[4])
        ts = float(items[5])

        self.strm_bytes.setdefault(strmid, {}).setdefault(circid, [])
        self.strm_bytes[strmid][circid].append([bw_bytes, direction, ts])

    def _handle_stream_event(self, items):
        chanid, circid, strmid, port, readbw, writebw = [long(v) for v in items[0:6]]
        start, end = float(items[6]), float(items[7])
        is_dns = True if int(items[8]) == 1 else False
        is_dir = True if int(items[9]) == 1 else False

        # only count streams with legitimate transfers
        totalbw = readbw+writebw
        if totalbw <= 0:
            return

        self.secure_counters.increment("StreamsAll", 1)
        self.secure_counters.increment("StreamBytesAll", 1, totalbw)

        self.circ_info.setdefault(chanid, {}).setdefault(circid, {'num_streams': {'interactive':0L, 'web':0L, 'p2p':0L, 'other':0L}, 'stream_starttimes': {'interactive':[], 'web':[], 'p2p':[], 'other':[]}})

        stream_class = self._classify_port(port)
        self.circ_info[chanid][circid]['num_streams'][stream_class] += 1L
        self.circ_info[chanid][circid]['stream_starttimes'][stream_class].append(start)

        # the amount we read from the stream is bound for the client
        # the amount we write to the stream is bound to the server
        ratio = self._encode_ratio(readbw, writebw)
        lifetime = end-start

        self.secure_counters.increment("StreamBytesOutAll", writebw)
        self.secure_counters.increment("StreamBytesInAll", readbw)
        self.secure_counters.increment("StreamBytesRatioAll", ratio)

        if stream_class == 'web':
            self.secure_counters.increment("StreamsWeb", 1)
            self.secure_counters.increment("StreamBytesWeb", 1, totalbw)
            self.secure_counters.increment("StreamBytesOutWeb", writebw)
            self.secure_counters.increment("StreamBytesInWeb", readbw)
            self.secure_counters.increment("StreamBytesRatioWeb", ratio)
            self.secure_counters.increment("StreamLifeTimeWeb", lifetime)
        elif stream_class == 'interactive':
            self.secure_counters.increment("StreamsInteractive", 1)
            self.secure_counters.increment("StreamBytesInteractive", 1, totalbw)
            self.secure_counters.increment("StreamBytesOutInteractive", writebw)
            self.secure_counters.increment("StreamBytesInInteractive", readbw)
            self.secure_counters.increment("StreamBytesRatioInteractive", ratio)
            self.secure_counters.increment("StreamLifeTimeInteractive", lifetime)
        elif stream_class == 'p2p':
            self.secure_counters.increment("StreamsP2P", 1)
            self.secure_counters.increment("StreamBytesP2P", 1, totalbw)
            self.secure_counters.increment("StreamBytesOutP2P", writebw)
            self.secure_counters.increment("StreamBytesInP2P", readbw)
            self.secure_counters.increment("StreamBytesRatioP2P", ratio)
            self.secure_counters.increment("StreamLifeTimeP2P", lifetime)
        elif stream_class == 'other':
            self.secure_counters.increment("StreamsOther", 1)
            self.secure_counters.increment("StreamBytesOther", 1, totalbw)
            self.secure_counters.increment("StreamBytesOutOther", writebw)
            self.secure_counters.increment("StreamBytesInOther", readbw)
            self.secure_counters.increment("StreamBytesRatioOther", ratio)
            self.secure_counters.increment("StreamLifeTimeOther", lifetime)

        # if we have a traffic model object, then we should use our observations to find the
        # most likely path through the HMM, and then count some aggregate statistics
        # about that path
        if self.traffic_model is not None and strmid in self.strm_bytes and circid in self.strm_bytes[strmid]:
            byte_events = self.strm_bytes[strmid][circid]
            observed_packet_delays = self._get_inter_packet_delays(start, byte_events)
            likliest_states = self.traffic_model.run_viterbi(observed_packet_delays)

            num_packets = len(observed_packet_delays)
            num_states = len(likliest_states)
            if num_states > num_packets:
                logging.warning("viterbi gave us more states than we have packets")
            elif num_states < num_packets:
                logging.warning("viterbi gave us fewer states than we have packets")

            '''
            now increment the counters that we will use to update the model

            example observations = [('+', 20), ('+', 10), ('+',50), ('+',1000)]
            example viterbi result: Blabbing Blabbing Blabbing Thinking
            then count the following 3:
              - increment 1 for state/observation match:
                Blabbing+, Blabbing+, Blabbing+, Thinking+
              - increment x where x is delay for each state/observation match:
                Blabbing+: 20, Blabbing+: 10, Blabbing+: 50, Thinking+: 1000
              - increment 1 for each state-to-state transition:
                BlabbingBlabbing, BlabbingBlabbing, BlabbingThinking
            '''

            for i in xrange(num_packets):
                (dir_code, delay) = observed_packet_delays[i] # delay is in microseconds
                state = likliest_states[i]

                self.secure_counters.increment("TrafficModelTotalEmissions", 1, num_increments=1)
                label = "TrafficModelTotalEmissions_{}{}".format(state, dir_code)
                self.secure_counters.increment(label, 1, num_increments=1)

                self.secure_counters.increment("TrafficModelTotalDelay", 1, num_increments=delay)
                label = "TrafficModelTotalDelay_{}{}".format(state, dir_code)
                self.secure_counters.increment(label, 1, num_increments=delay)

                if (i+1) < num_states:
                    next_state = likliest_states[i+1]
                    self.secure_counters.increment("TrafficModelTotalTransitions", 1, num_increments=1)
                    label = "TrafficModelTotalTransitions_{}_{}".format(state, next_state)
                    self.secure_counters.increment(label, 1, num_increments=1)

        # clear all 'packet' data for this stream
        if strmid in self.strm_bytes:
            self.strm_bytes[strmid].pop(circid, None)
            if len(self.strm_bytes[strmid]) == 0:
                self.strm_bytes.pop(strmid, None)

    def _get_inter_packet_delays(self, strm_start_ts, byte_events):
        '''
        Take a list of (bw_bytes, direction, ts) and turn them into packet delay
        observations of the form [('+', 20), ('+', 10), ('+',50), ('+',1000)]
          - the returned delays will be in microseconds
          - the returned list of observations is suitable to pass to
            TrafficModel.run_viterbi() to find the most likly path (series of states)
            through our traffic model
        '''
        packet_delays = []
        num_events = len(byte_events)
        for i in xrange(num_events):
            (bw_bytes, direction, ts) = byte_events[i]

            # a certain number of bytes were read from the kernel, turn these into packets
            dir_code = '+' if direction == "outbound" else '-' # '-' for direction == "inbound"
            # ts is unix timestamp in sec.microsec, like 12345678.123456
            # so delay will be in microseconds
            seconds_since_stream_start = ts - strm_start_ts
            micros = seconds_since_stream_start * 1000000
            delay = max(long(0), long(micros))

            # create a packet delay for each packet
            while bw_bytes > 0:
                packet_delays.append((dir_code, delay))
                # the first packet gets all of the delay, the others arrive at the same time
                delay = 0
                bw_bytes -= 1500 # MTU
        return packet_delays

    def _classify_port(self, port):
        p2p_ports = [1214]
        for p in xrange(4661, 4666+1): p2p_ports.append(p)
        for p in xrange(6346, 6429+1): p2p_ports.append(p)
        p2p_ports.append(6699)
        for p in xrange(6881, 6999+1): p2p_ports.append(p)

        if port in [80, 443]:
            return 'web'
        elif port in [22, 194, 994, 6660, 6661, 6662, 6663, 6664, 6665, 6666, 6667, 6668, 6669, 6670, 6679, 6697, 7000]:
            return 'interactive'
        elif port in p2p_ports:
            return 'p2p'
        else:
            return 'other'

    def _encode_ratio(self, inval, outval):
        if inval == outval:
            return 0.0
        elif inval == 0.0:
            return float('inf')
        elif outval == 0.0:
            return float('-inf')
        else:
            return math.log(float(outval)/float(inval), 2) # log base 2

    def _compute_interstream_creation_times(self, l):
        l.sort()
        times = []
        for i in xrange(len(l)):
            if i == 0: continue
            times.append(l[i] - l[i-1])
        return times

    def _handle_circuit_event(self, items):
        chanid, circid, ncellsin, ncellsout, readbwdns, writebwdns, readbwexit, writebwexit = [long(v) for v in items[0:8]]
        start, end = float(items[8]), float(items[9])
        previp = items[10]
        prevIsClient = True if int(items[11]) > 0 else False
        prevIsRelay = True if int(items[12]) > 0 else False
        nextip = items[13]
        nextIsClient = True if int(items[14]) > 0 else False
        nextIsRelay = True if int(items[15]) > 0 else False

        # we get circuit events on both exits and entries
        # stream bw info is only avail on exits
        # isclient is based on CREATE_FAST and I'm not sure that is always used by clients
        if not prevIsRelay:
            # previous hop is unkown, we are entry
            self.secure_counters.increment("CircuitsAllEntry", 1)

            # only count cells ratio on active circuits with legitimate transfers
            is_active = True if ncellsin + ncellsout >= 8 else False
            if is_active:
                self.secure_counters.increment("CircuitsActiveEntry", 1)
                self.secure_counters.increment("CircuitCellsIn", ncellsin)
                self.secure_counters.increment("CircuitCellsOut", ncellsout)
                self.secure_counters.increment("CircuitCellsRatio", self._encode_ratio(ncellsin, ncellsout))
            else:
                self.secure_counters.increment("CircuitsInactiveEntry", 1)

            # count unique client ips
            # we saw this client within current rotation window
            self.cli_ips_current.setdefault(previp, {'is_active':False})
            if is_active:
                self.cli_ips_current[previp]['is_active'] = True
            if start < self.cli_ips_rotated:
                # we also saw the client in the previous rotation window
                self.cli_ips_previous.setdefault(previp, {'is_active':False})
                if is_active:
                    self.cli_ips_previous[previp]['is_active'] = True

            # count number of completed circuits per client
            if is_active:
                if 'num_active_completed' not in self.cli_ips_current[previp]:
                    self.cli_ips_current[previp]['num_active_completed'] = 0L
                self.cli_ips_current[previp]['num_active_completed'] += 1L
            else:
                if 'num_inactive_completed' not in self.cli_ips_current[previp]:
                    self.cli_ips_current[previp]['num_inactive_completed'] = 0L
                self.cli_ips_current[previp]['num_inactive_completed'] += 1L

        elif not nextIsRelay:
            # prev hop is known relay but next is not, we are exit
            self.secure_counters.increment("CircuitsAll", 1)

            # check if we have any stream info in this circuit
            circ_is_known, has_completed_stream = False, False
            if chanid in self.circ_info and circid in self.circ_info[chanid]:
                circ_is_known = True
                if sum(self.circ_info[chanid][circid]['num_streams'].values()) > 0:
                    has_completed_stream = True

            if circ_is_known and has_completed_stream:
                # we have circuit info and at least one stream ended on it
                self.secure_counters.increment("CircuitsActive", 1)
                self.secure_counters.increment("CircuitLifeTime", end - start)

                # convenience
                counts = self.circ_info[chanid][circid]['num_streams']
                times = self.circ_info[chanid][circid]['stream_starttimes']

                # first increment general counters
                self.secure_counters.increment("CircuitStreamsAll", sum(counts.values()))
                for isct in self._compute_interstream_creation_times(times['web'] + times['interactive'] + times['p2p'] + times['other']):
                    self.secure_counters.increment("CircuitInterStreamCreationTime", isct)

                # now only increment the classes that have positive counts
                if counts['web'] > 0:
                    self.secure_counters.increment("CircuitsWeb", 1)
                    self.secure_counters.increment("CircuitStreamsWeb", counts['web'])
                    for isct in self._compute_interstream_creation_times(times['web']):
                        self.secure_counters.increment("CircuitInterStreamCreationTimeWeb", isct)
                if counts['interactive'] > 0:
                    self.secure_counters.increment("CircuitsInteractive", 1)
                    self.secure_counters.increment("CircuitStreamsInteractive", counts['interactive'])
                    for isct in self._compute_interstream_creation_times(times['interactive']):
                        self.secure_counters.increment("CircuitInterStreamCreationTimeInteractive", isct)
                if counts['p2p'] > 0:
                    self.secure_counters.increment("CircuitsP2P", 1)
                    self.secure_counters.increment("CircuitStreamsP2P", counts['p2p'])
                    for isct in self._compute_interstream_creation_times(times['p2p']):
                        self.secure_counters.increment("CircuitInterStreamCreationTimeP2P", isct)
                if counts['other'] > 0:
                    self.secure_counters.increment("CircuitsOther", 1)
                    self.secure_counters.increment("CircuitStreamsOther", counts['other'])
                    for isct in self._compute_interstream_creation_times(times['other']):
                        self.secure_counters.increment("CircuitInterStreamCreationTimeOther", isct)

            else:
                # either we dont know circ, or no streams ended on it
                self.secure_counters.increment("CircuitsInactive", 1)

            # cleanup
            if circ_is_known:
                # remove circ from channel
                self.circ_info[chanid].pop(circid, None)
                # if that was the last circuit on channel, remove the channel too
                if len(self.circ_info[chanid]) == 0:
                    self.circ_info.pop(chanid, None)

    def _handle_connection_event(self, items):
        chanid = long(items[0])
        start, end = float(items[1]), float(items[2])
        ip = items[3]
        isclient = True if int(items[4]) > 0 else False
        isrelay = True if int(items[5]) > 0 else False
        if not isrelay:
            self.secure_counters.increment("ConnectionsAll", 1)
            self.secure_counters.increment("ConnectionLifeTime", end - start)

    def _do_rotate(self):
        logging.info("rotating circuit window now, {}".format(format_last_event_time_since(self.last_event_time)))

        # dont count anything in the first rotation period, since events that ended up in the
        # previous list will be skewed torward longer lived circuits
        if True:#self.num_rotations > 0:
            for ip in self.cli_ips_previous:
                client = self.cli_ips_previous[ip]

                self.secure_counters.increment("ClientIPsUnique", 1)
                if client['is_active']:
                    self.secure_counters.increment("ClientIPsActive", 1)
                else:
                    self.secure_counters.increment("ClientIPsInactive", 1)

                if 'num_active_completed' in client:
                    self.secure_counters.increment("ClientIPCircuitsActive", client['num_active_completed'])
                if 'num_inactive_completed' in client:
                    self.secure_counters.increment("ClientIPCircuitsInactive", client['num_inactive_completed'])

        # reset for next interval
        self.cli_ips_previous = self.cli_ips_current
        self.cli_ips_current = {}
        self.cli_ips_rotated = time()
        self.num_rotations += 1L
