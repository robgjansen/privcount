# See LICENSE for licensing information

import os
import logging
import math
import string
import sys
import cPickle as pickle
import yaml

from time import time
from copy import deepcopy
from base64 import b64decode

from twisted.internet import task, reactor, ssl
from twisted.internet.protocol import ReconnectingClientFactory

from privcount.config import normalise_path, choose_secret_handshake_path
from privcount.connection import connect, disconnect, validate_connection_config, choose_a_connection, get_a_control_password
from privcount.counter import SecureCounters, counter_modulus, add_counter_limits_to_config, combine_counters, has_noise_weight, get_noise_weight, count_bins, are_events_expected, get_valid_counters
from privcount.crypto import get_public_digest_string, load_public_key_string, encrypt
from privcount.log import log_error, format_delay_time_wait, format_last_event_time_since, format_elapsed_time_since, errorCallback, summarise_string
from privcount.node import PrivCountClient, EXPECTED_EVENT_INTERVAL_MAX, EXPECTED_CONTROL_ESTABLISH_MAX
from privcount.protocol import PrivCountClientProtocol, TorControlClientProtocol, get_privcount_version
from privcount.tagged_event import parse_tagged_event, is_string_valid, is_list_valid, is_int_valid, is_flag_valid, is_float_valid, is_ip_address_valid, get_string_value, get_list_value, get_int_value, get_flag_value, get_float_value, get_ip_address_value
from privcount.traffic_model import TrafficModel, check_traffic_model_config

SINGLE_BIN = SecureCounters.SINGLE_BIN

# using reactor: pylint: disable=E1101
# method docstring missing: pylint: disable=C0111
# line too long: pylint: disable=C0301

class DataCollector(ReconnectingClientFactory, PrivCountClient):
    '''
    receive key share data from the DC message receiver
    keep the shares during collection epoch
    send the shares to the TS at end of epoch
    '''

    def __init__(self, config_filepath):
        PrivCountClient.__init__(self, config_filepath)
        self.aggregator = None
        self.is_aggregator_pending = False
        self.context = {}
        self.expected_aggregator_start_time = None

    def buildProtocol(self, addr):
        '''
        Called by twisted
        '''
        return PrivCountClientProtocol(self)

    def startFactory(self):
        '''
        Called by twisted
        '''
        # TODO
        return
        state = self.load_state()
        if state is not None:
            self.aggregator = state['aggregator']
            self.is_aggregator_pending = state['is_aggregator_pending']

    def stopFactory(self):
        '''
        Called by twisted
        '''
        # TODO
        return
        if self.aggregator is not None:
            # export everything that would be needed to survive an app restart
            state = {'aggregator': self.aggregator, 'is_aggregator_pending': self.is_aggregator_pending}
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

        # connect to the tally server, register, and wait for commands
        self.do_checkin()
        reactor.run()

    def get_status(self):
        '''
        Called by protocol
        Returns a dictionary containing status information
        '''
        status = {
            'type' : 'DataCollector',
            'name' : self.config['name'],
            'state' : 'active' if self.aggregator is not None else 'idle',
            'privcount_version' : get_privcount_version(),
                 }
        # store the latest context, so we have it even when the aggregator goes away
        if self.aggregator is not None:
            self.context.update(self.aggregator.get_context())
        # and include the latest context values in the status
        status.update(self.context)
        return status

    def get_flag_list(self):
        '''
        Return the flags for our relay in the most recent consensus.
        If there are no flags or no consensus entry or no context,
        return an empty list.
        '''
        return self.context.get('flag_list', [])

    def get_counter_list(self):
        '''
        Return the list of counters we are collecting, or None if we haven't
        started yet.
        '''
        if self.start_config is None:
            return None
        else:
            # we should have caught unknown counters when we started
            return self.check_start_config(self.start_config,
                                           allow_unknown_counters=False)

    def are_dc_events_expected(self):
        '''
        Return True if we expect to receive events regularly.
        Return False if we don't.
        '''
        dc_counters = self.get_counter_list()
        flag_list = self.get_flag_list()
        return are_events_expected(dc_counters, flag_list)

    def check_aggregator(self):
        '''
        If the aggregator is live, but isn't getting events, log a diagnostic
        warning.
        This function is sometimes called using deferLater, so any exceptions
        will be handled by errorCallback.
        '''
        if (self.aggregator is not None and not self.is_aggregator_pending and
            self.expected_aggregator_start_time is not None and
            self.expected_aggregator_start_time < time()):
            aggregator_live_time = time() - self.expected_aggregator_start_time
            flag_message = "Is your relay in the Tor consensus?"
            flag_list = self.get_flag_list()
            if len(flag_list) > 0:
                flag_message = "Consensus flags: {}".format(" ".join(flag_list))
            if self.are_dc_events_expected():
                log_fn = logging.warning
            else:
                log_fn = logging.info
            if ((self.aggregator.protocol is None or
                 self.aggregator.protocol.state != "processing") and
                aggregator_live_time > EXPECTED_CONTROL_ESTABLISH_MAX):
                logging.warning("Aggregator has been running {}, but is not connected to the control port. Is your control port working?"
                                .format(format_elapsed_time_since(
                                        self.expected_aggregator_start_time,
                                        'since')))
            elif (self.aggregator.last_event_time is None and
                  aggregator_live_time > EXPECTED_EVENT_INTERVAL_MAX):
                log_fn("Aggregator has been running {}, but has not seen a tor event. {}"
                       .format(format_elapsed_time_since(
                                          self.expected_aggregator_start_time,
                                          'since'),
                               flag_message))
            elif (self.aggregator.last_event_time is not None and
                  self.aggregator.last_event_time < time() - EXPECTED_EVENT_INTERVAL_MAX):
                log_fn("Aggregator has not received any events recently, {}. {}"
                       .format(format_last_event_time_since(
                                             self.aggregator.last_event_time),
                               flag_message))

    def do_checkin(self):
        '''
        Called by protocol
        Refresh the config, and try to connect to the server
        This function is usually called using LoopingCall, so any exceptions
        will be turned into log messages.
        '''
        # TODO: Refactor common client code - issue #121
        self.refresh_config()
        self.check_aggregator()

        ts_ip = self.config['tally_server_info']['ip']
        ts_port = self.config['tally_server_info']['port']
        # turn on reconnecting mode and reset backoff
        self.resetDelay()
        logging.info("checking in with TallyServer at {}:{}".format(ts_ip, ts_port))
        reactor.connectSSL(ts_ip, ts_port, self, ssl.ClientContextFactory()) # pylint: disable=E1101

    def do_start(self, config):
        '''
        this is called by the protocol when we receive a command from the TS
        to start a new collection phase
        return None if failure, otherwise json will encode result
        '''
        # keep the start config to send to the TS at the end of the collection
        # deepcopy in case we make any modifications later
        self.start_config = deepcopy(config)

        if ('sharekeepers' not in config):
            logging.warning("start command from tally server cannot be completed due to missing sharekeepers")
            return None

        dc_counters = self.check_start_config(config,
                                              allow_unknown_counters=False)

        if dc_counters is None:
            return None

        # if we are still running from a previous incarnation, we need to stop
        # first
        if self.aggregator is not None:
            return None

        # we require that only the configured share keepers be used in the
        # collection phase, because we must be able to encrypt messages to them
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

        # if we got a traffic model from the tally server and it passes validation,
        # then load the traffic model object that we will use during aggregation
        traffic_model_config = None
        if 'traffic_model' in config:
            traffic_model_config = config['traffic_model']

        # The aggregator doesn't care about the DC threshold
        self.aggregator = Aggregator(dc_counters,
                                     traffic_model_config,
                                     config['sharekeepers'],
                                     config['noise_weight'],
                                     counter_modulus(),
                                     self.config['event_source'],
                                     self.config['rotate_period'],
                                     self.config['use_setconf'],
                                     config.get('max_cell_events_per_circuit', -1),
                                     config.get('circuit_sample_rate', 1.0))

        defer_time = config['defer_time'] if 'defer_time' in config else 0.0
        logging.info("got start command from tally server, starting aggregator in {}".format(format_delay_time_wait(defer_time, 'at')))
        self.expected_aggregator_start_time = time() + defer_time

        # sync the time that we start listening for Tor events
        self.is_aggregator_pending = True
        aggregator_deferred = task.deferLater(reactor, defer_time,
                                              self._start_aggregator_deferred)
        aggregator_deferred.addErrback(errorCallback)
        # return the generated shares now
        shares = self.aggregator.get_shares()
        # this is a dict {sk_uid : sk_msg} for each sk
        for sk_uid in shares:
            # add the sender's name for debugging purposes
            shares[sk_uid]['dc_name'] = self.config['name']
            # encrypt shares[sk_uid] for that sk
            pub_key_str = b64decode(config['sharekeepers'][sk_uid])
            sk_pub_key = load_public_key_string(pub_key_str)
            encrypted_secret = encrypt(sk_pub_key, shares[sk_uid]['secret'])
            # TODO: secure delete
            shares[sk_uid]['secret'] = encrypted_secret

        logging.info("successfully started and generated {} blinding shares for {} counters ({} bins)"
                     .format(len(shares), len(dc_counters), count_bins(dc_counters)))
        return shares

    def _start_aggregator_deferred(self):
        '''
        This function is called using deferLater, so any exceptions will be
        handled by errorCallback.
        '''
        if self.is_aggregator_pending:
            self.is_aggregator_pending = False
            self.aggregator.start()
            # schedule a once-off check that the aggregator has connected
            check_aggregator_deferred = task.deferLater(
                                            reactor,
                                            EXPECTED_CONTROL_ESTABLISH_MAX + 1.0,
                                            self.check_aggregator)
            check_aggregator_deferred.addErrback(errorCallback)

    def do_stop(self, config):
        '''
        called by protocol
        the TS wants us to stop the current collection phase
        they may or may not want us to send back our counters
        stop the node from running
        return a dictionary containing counters (if available and wanted)
        and the local and start configs
        '''
        logging.info("got command to stop collection phase")

        counts = None
        if self.aggregator is not None and not self.is_aggregator_pending:
            counts = self.aggregator.stop()

        if self.aggregator is not None:
            # TODO: secure delete
            del self.aggregator
            self.aggregator = None
        else:
            logging.info("No aggregator, counts never started")

        if self.is_aggregator_pending:
            self.is_aggregator_pending = False
            logging.info("Aggregator deferred, counts never started")

        self.expected_aggregator_start_time = None

        return self.check_stop_config(config, counts)

    DEFAULT_ROTATE_PERIOD = 600

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
            dc_conf = conf['data_collector']

            # find the path for the secret handshake file
            dc_conf['secret_handshake'] = choose_secret_handshake_path(
                dc_conf, conf)

            # the state file (unused)
            if 'state' in dc_conf:
                del dc_conf['state']
            #dc_conf['state'] = normalise_path(dc_conf['state'])
            #assert os.path.exists(os.path.dirname(dc_conf['state']))

            dc_conf['delay_period'] = self.get_valid_delay_period(dc_conf)

            dc_conf.setdefault('always_delay', False)
            assert isinstance(dc_conf['always_delay'], bool)

            dc_conf['rotate_period'] = dc_conf.get('rotate_period',
                                          conf.get('rotate_period',
                                                   DataCollector.DEFAULT_ROTATE_PERIOD))
            assert dc_conf['rotate_period'] > 0

            # Data collectors use SETCONF by default
            dc_conf.setdefault('use_setconf', True)
            dc_conf['use_setconf'] = bool(dc_conf['use_setconf'])

            dc_conf['sigma_decrease_tolerance'] = \
                self.get_valid_sigma_decrease_tolerance(dc_conf)

            assert dc_conf['name'] != ''

            assert validate_connection_config(dc_conf['tally_server_info'],
                                              must_have_ip=True)
            assert validate_connection_config(dc_conf['event_source'])

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

    def __init__(self, counters, traffic_model_config, sk_uids,
                 noise_weight, modulus, tor_control_port, rotate_period,
                 use_setconf, max_cell_events_per_circuit, circuit_sample_rate):
        self.secure_counters = SecureCounters(counters, modulus,
                                              require_generate_noise=True)
        self.collection_counters = counters
        # we can't generate the noise yet, because we don't know the
        # DC fingerprint
        self.secure_counters.generate_blinding_shares(sk_uids)

        # the traffic model is optional
        self.traffic_model = None
        if traffic_model_config is not None:
            self.traffic_model = TrafficModel(traffic_model_config)

        self.noise_weight_config = noise_weight
        self.noise_weight_value = None
        self.max_cell_events_per_circuit = int(max_cell_events_per_circuit)
        self.circuit_sample_rate = float(circuit_sample_rate)

        self.connector = None
        self.connector_list = None
        self.protocol = None
        self.rotator = None

        self.tor_control_port = tor_control_port
        self.rotate_period = rotate_period
        self.use_setconf = use_setconf

        self.last_event_time = None
        self.num_rotations = 0
        self.circ_info = {}
        self.strm_bytes = {}
        self.cli_ips_rotated = time()
        self.cli_ips_current = {}
        self.cli_ips_previous = {}

        self.nickname = None
        self.orport_list = []
        self.dirport_list = []
        self.tor_version = None
        self.tor_privcount_version = None
        self.address = None
        self.fingerprint = None
        self.flag_list = []

    def buildProtocol(self, addr):
        if self.protocol is not None:
            if self.protocol.isConnected():
                logging.info('Request for existing protocol: returning existing connected procotol')
                return self.protocol
            else:
                logging.info('Request for existing protocol: deleting disconnected protocol and returning new procotol')
                self.protocol.clearConnection('build procotol')
        else:
            logging.debug('Request for new protocol: returning new procotol')
        self.protocol = TorControlClientProtocol(self)
        # if we didn't build the protocol until after starting
        if self.connector is not None:
            self.protocol.startCollection(self.collection_counters)
        return self.protocol

    def startFactory(self):
        # TODO
        return

    def stopFactory(self):
        # TODO
        return

    def start(self):
        '''
        start the aggregator, and connect to the control port
        '''
        # This call can return a list of connectors, or a single connector
        self.connector_list = connect(self, self.tor_control_port)
        # Twisted doesn't want a list of connectors, it only wants one
        self.connector = choose_a_connection(self.connector_list)
        self.rotator = task.LoopingCall(self._do_rotate)
        rotator_deferred = self.rotator.start(self.rotate_period, now=False)
        rotator_deferred.addErrback(errorCallback)
        # if we've already built the protocol before starting
        if self.protocol is not None:
            self.protocol.startCollection(self.collection_counters)

    def _stop_protocol(self):
        '''
        Stop protocol and connection activities.
        '''
        # don't try to reconnect
        self.stopTrying()

        # stop reading from Tor control port
        if self.protocol is not None:
            self.protocol.stopCollection()
            self.protocol.quit()
            self.protocol = None
        if self.rotator is not None and self.rotator.running:
            self.rotator.stop()
            self.rotator = None
        if self.connector_list is not None:
            disconnect(self.connector_list)
            self.connector_list = None
            self.connector = None

    def _stop_secure_counters(self, counts_are_valid=True):
        '''
        If counts_are_valid, detach and return the counts from secure counters.
        Otherwise, return None.
        '''
        # if we've already stopped counting due to an error, there are no
        # counters
        if self.secure_counters is None:
            return None

        # return the final counts and make sure we cant be restarted
        counts = self.secure_counters.detach_counts()
        # TODO: secure delete?
        del self.secure_counters
        self.secure_counters = None
        if counts_are_valid:
            return counts
        else:
            return None

    def stop(self, counts_are_valid=True):
        '''
        Stop counting, and stop connecting to the ControlPort and Tally Server.
        Retrieve the counts, and delete the counters.
        If counts_are_valid is True, return the counts.
        Otherwise, return None.
        '''
        # make sure we added noise
        if self.noise_weight_value is None and counts_are_valid:
            logging.warning("Noise was not added to counters when the control port connection was opened. Adding now.")
            self.generate_noise()

        # stop trying to collect data
        self._stop_protocol()

        # stop using the counters
        return self._stop_secure_counters(counts_are_valid=counts_are_valid)

    def get_shares(self):
        return self.secure_counters.detach_blinding_shares()

    def generate_noise(self):
        '''
        If self.fingerprint is included in the noise weight config from the
        tally server, add noise to the counters based on the weight for that
        fingerprint.
        If not, stop participating in the round and delete all counters.
        Must be called before detaching counters.
        '''
        if self.noise_weight_value is not None:
            logging.warning("Asked to add noise twice. Ignoring.")
            return

        if has_noise_weight(self.noise_weight_config, self.fingerprint):
            self.noise_weight_value = get_noise_weight(
                self.noise_weight_config, self.fingerprint)
            self.secure_counters.generate_noise(self.noise_weight_value)
        else:
            logging.warning("Tally Server did not provide a noise weight for our fingerprint {} in noise weight config {}, we will not count in this round."
                            .format(self.fingerprint,
                                    self.noise_weight_config))
            # stop collecting and stop counting
            self._stop_protocol()
            self._stop_secure_counters(counts_are_valid=False)

    def get_control_password(self):
        '''
        Return the configured control password for this data collector, or
        None if no connections have a control password.
        '''
        # Multiple different control passwords are not supported
        return get_a_control_password(self.tor_control_port)

    def get_use_setconf(self):
        '''
        Returns True if the protocol should use SETCONF, and False if it should
        rely on the torrc or another PrivCount instance to set EnablePrivCount.
        '''
        return self.use_setconf

    def get_max_cell_events_per_circuit(self):
        '''
        Returns the PrivCountMaxCellEventsPerCircuit, an integer number in
        [-INT_MAX, INT_MAX]. Defaults to -1 (send all cell events for each circuit).
        '''
        # check that the value is valid
        assert self.max_cell_events_per_circuit is not None
        # check that we can apply non-default values
        if (not self.get_use_setconf() and
            self.max_cell_events_per_circuit >= 0):
            logging.warning("PrivCountMaxCellEventsPerCircuit {} ignored because use_setconf is False."
                            .format(self.max_cell_events_per_circuit))
        return self.max_cell_events_per_circuit

    def get_circuit_sample_rate(self):
        '''
        Returns the PrivCountCircuitSampleRate, a floating-point number in
        [0.0, 1.0]. Defaults to 1.0 (send events for all circuits, streams,
        and cells).
        '''
        # check that the value is valid
        assert self.circuit_sample_rate is not None
        assert self.circuit_sample_rate >= 0.0
        assert self.circuit_sample_rate <= 1.0
        # check that we can apply non-default values
        if (not self.get_use_setconf() and
            self.circuit_sample_rate != 1.0):
            logging.warning("PrivCountCircuitSampleRate {} ignored because use_setconf is False."
                            .format(self.circuit_sample_rate))
        return self.circuit_sample_rate

    def set_nickname(self, nickname):
        nickname = nickname.strip()

        # Do some basic validation of the nickname
        if len(nickname) > 19:
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

    @staticmethod
    def validate_tor_port(tor_port, description):
        '''
        Validate a single Tor ORPort or DirPort entry, using description as
        the port type in any log messages.
        tor_port is an ORPort or DirPort config line.
        Some can be IPv6 *Ports, which have an IPv6 address and a port.
        Others include options, such as NoListen.
        '''
        # Do some basic validation of the port
        # There isn't much we can do here, because port lines vary so much
        if len(tor_port) < 1 or len(tor_port) > 200:
            logging.warning("Bad %s length %d: %s",
                          description, len(tor_port), tor_port)
            return False
        if not all(c in string.printable for c in tor_port):
            logging.warning("Bad %s characters: %s", description, tor_port)
            return False
        return True

    @staticmethod
    def add_tor_port(tor_port, tor_port_list, description):
        '''
        Add a single Tor ORPort or DirPort entry to tor_port_list, using
        description as the port type in any log messages.
        '''
        if tor_port in tor_port_list:
            logging.info("Ignoring duplicate %s: %s", description, tor_port)
        else:
            tor_port_list.append(tor_port)
            tor_port_list.sort()

    @staticmethod
    def get_tor_port(tor_port_list, description):
        '''
        Create a list of all known *Ports on the relay from tor_port_list,
        using description as the port type in any log messages.
        '''
        if len(tor_port_list) == 0:
            return None
        else:
            return ", ".join(tor_port_list)

    def set_orport(self, orport):
        '''
        Add an ORPort to the set of ORPorts on the relay.
        A relay can have multiple ORPorts.
        See validate_tor_port for how ORPorts are validated.
        '''
        orport = orport.strip()

        if not Aggregator.validate_tor_port(orport, 'ORPort'):
            return False

        Aggregator.add_tor_port(orport, self.orport_list, 'ORPort')

        return True

    def get_orport(self):
        '''
        Get a comma-separated list of ORPorts on the relay.
        '''
        return Aggregator.get_tor_port(self.orport_list, 'ORPort')

    def set_dirport(self, dirport):
        '''
        Like set_orport, but for DirPorts.
        '''
        dirport = dirport.strip()

        if not Aggregator.validate_tor_port(dirport, 'DirPort'):
            return False

        Aggregator.add_tor_port(dirport, self.dirport_list, 'DirPort')

        return True

    def get_dirport(self):
        '''
        Like get_orport, but for DirPorts.
        '''
        return Aggregator.get_tor_port(self.dirport_list, 'DirPort')

    @staticmethod
    def validate_version(version, old_version, description):
        '''
        Perform basic validation and processing on version.
        Uses description for logging changes to old_version.
        Returns a whitespace-stripped version string, or None if the version
        is invalid.
        '''
        if "version" in version:
            _, _, version = version.partition("version")
        version = version.strip()

        # Do some basic validation of the version
        # This is hard, because versions can be almost anything
        if not len(version) > 0:
            logging.warning("Bad %s version length %d: %s",
                            description, len(version), version)
            return None
        # This means unicode printables, there's no ASCII equivalent
        if not all(c in string.printable for c in version):
            logging.warning("Bad %s version characters: %s",
                            description, version)
            return None

        # Are we replacing an existing version?
        if old_version is not None:
            if old_version != version:
                if old_version.lower() in version.lower():
                    # we just added a git tag to the version
                    # this happens because GETINFO version has the tag, but
                    # PROTOCOLINFO does not
                    logging_level = logging.debug
                else:
                    # did someone just restart tor with a new version?
                    logging_level = logging.warning
                logging_level("Replacing %s version %s with %s",
                              description, old_version, version)
            else:
                logging.debug("Duplicate %s version received %s",
                              description, version)
        return version

    def set_tor_version(self, version):
        validated_version = Aggregator.validate_version(version, self.tor_version,
                                                        'Tor version')
        if validated_version is not None:
            self.tor_version = validated_version
            logging.info("Tor version is {}".format(self.tor_version))
            return True
        else:
            return False

    def get_tor_version(self):
        return self.tor_version

    def set_tor_privcount_version(self, tor_privcount_version):
      validated_version = Aggregator.validate_version(
                                                tor_privcount_version,
                                                self.tor_privcount_version,
                                                'Tor PrivCount version')
      if validated_version is not None:
          self.tor_privcount_version = validated_version
          logging.info("Tor PrivCount version is {}"
                       .format(self.tor_privcount_version))
          return True
      else:
          return False

    def get_tor_privcount_version(self):
        return self.tor_privcount_version

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
        '''
        If fingerprint is valid, set our stored fingerprint to fingerprint, and
        return True.
        Otherwise, return False.
        Called by TorControlClientProtocol.
        '''
        fingerprint = fingerprint.strip()

        # Do some basic validation of the fingerprint
        if not len(fingerprint) == 40:
            logging.warning("Bad fingerprint length %d: %s", len(fingerprint), fingerprint)
            return False
        if not all(c in string.hexdigits for c in fingerprint):
            logging.warning("Bad fingerprint characters: %s", fingerprint)
            return False

        # Is this the first time we've been told a fingerprint?
        if self.fingerprint is None:
            self.fingerprint = fingerprint
            self.generate_noise()
        else:
            if self.fingerprint != fingerprint:
                logging.warning("Received different fingerprint %s, keeping original fingerprint %s",
                                self.fingerprint, fingerprint)
            else:
                logging.debug("Duplicate fingerprint received %s", fingerprint)

        return True

    def get_fingerprint(self):
        '''
        Return the stored fingerprint for this relay.
        '''
        return self.fingerprint

    def set_flag_list(self, flag_string):
        '''
        Set our stored flag list to the list of space-separated flags in
        fingerprint. Ignores flag_string if it is None.
        Always returns True.
        Called by TorControlClientProtocol.
        '''
        if flag_string is None:
            logging.warning("flag_string was None in set_flag_list()")
            return True

        self.flag_list = flag_string.split()
        logging.info("Updated relay flags to {}".format(self.flag_list))

        return True

    def get_flag_list(self):
        '''
        Return the stored flag list for this relay.
        '''
        return self.flag_list

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
        if self.get_tor_version() is not None:
            context['tor_version'] = self.get_tor_version()
        if self.get_tor_privcount_version() is not None:
            context['tor_privcount_version'] = self.get_tor_privcount_version()
        if self.get_address() is not None:
            context['address'] = self.get_address()
        if self.get_fingerprint() is not None:
            context['fingerprint'] = self.get_fingerprint()
        if self.get_flag_list() is not None:
            context['flag_list'] = self.get_flag_list()
        if self.last_event_time is not None:
            context['last_event_time'] = self.last_event_time
        if self.noise_weight_value is not None:
            context['noise_weight_value'] = self.noise_weight_value
        return context

    def handle_event(self, event):
        if not self.secure_counters:
            return False

        # fail on events with no code
        if len(event) < 1:
            return False

        event_code, items = event[0], event[1:]
        self.last_event_time = time()

        # hand valid events off to the aggregator
        # keep events in order of frequency, particularly the cell and bytes
        # events (cell happens every 514 bytes, bytes happens every ~16kB)

        # This event has tagged fields: fields may be optional.
        if event_code == 'PRIVCOUNT_CIRCUIT_CELL':
            return self._handle_tagged_event(event_code, items)

        # these events have positional fields: order matters
        elif event_code == 'PRIVCOUNT_STREAM_BYTES_TRANSFERRED':
            if len(items) == Aggregator.STREAM_BYTES_ITEMS:
                return self._handle_bytes_event(items[:Aggregator.STREAM_BYTES_ITEMS])
            else:
                logging.warning("Rejected malformed {} event"
                                .format(event_code))
                return False

        elif event_code == 'PRIVCOUNT_STREAM_ENDED':
            if len(items) == Aggregator.STREAM_ENDED_ITEMS:
                return self._handle_stream_event(items[:Aggregator.STREAM_ENDED_ITEMS])
            else:
                logging.warning("Rejected malformed {} event"
                                .format(event_code))
                return False

        elif event_code == 'PRIVCOUNT_CIRCUIT_ENDED':
            if len(items) == Aggregator.CIRCUIT_ENDED_ITEMS:
                # Since 1.1.0, this legacy event is ignored, and the
                # fields are taken from PRIVCOUNT_CIRCUIT_CLOSE
                return True
            else:
                # Warn, but don't quit the connection
                logging.warning("Rejected malformed {} event"
                                .format(event_code))
                return True

        elif event_code == 'PRIVCOUNT_CONNECTION_ENDED':
            if len(items) == Aggregator.CONNECTION_ENDED_ITEMS:
                return self._handle_connection_event(items[:Aggregator.CONNECTION_ENDED_ITEMS])
            else:
                logging.warning("Rejected malformed {} event"
                                .format(event_code))
                return False
        # These events have tagged fields: fields may be optional.
        else:
            return self._handle_tagged_event(event_code, items)

        # TODO: secure delete
        #del items
        #del fields

        return True

    def _handle_tagged_event(self, event_code, items):
        '''
        Handle an event with tagged fields.
        '''

        if (event_code == 'PRIVCOUNT_CIRCUIT_CELL' or
            event_code == 'PRIVCOUNT_CIRCUIT_CLOSE' or
            event_code == 'PRIVCOUNT_HSDIR_CACHE_STORE'):
            fields = parse_tagged_event(items)
        else:
            logging.warning("Unexpected {} event when parsing: '{}'"
                            .format(event_code, " ".join(items)))
            return True

        # malformed: handle by warning and ignoring
        if len(items) > 0 and len(fields) == 0:
            logging.warning("Ignored malformed {} event: '{}'"
                          .format(event_code, " ".join(items)))
            return True
        elif event_code == 'PRIVCOUNT_CIRCUIT_CELL':
            return self._handle_circuit_cell_event(fields)
        elif event_code == 'PRIVCOUNT_CIRCUIT_CLOSE':
            return self._handle_circuit_close_event(fields)
        elif event_code == 'PRIVCOUNT_HSDIR_CACHE_STORE':
            return self._handle_hsdir_stored_event(fields)
        else:
            logging.warning("Unexpected {} event when handling: '{}'"
                            .format(event_code, " ".join(items)))
            return True
        return True

    STREAM_BYTES_ITEMS = 6

    # Positional event: fields is a list of Values.
    # All fields are mandatory, order matters
    # 'PRIVCOUNT_STREAM_BYTES_TRANSFERRED', ChanID, CircID, StreamID, isOutbound, BW, Time
    # See doc/TorEvents.markdown for details
    def _handle_bytes_event(self, items):
        assert(len(items) == Aggregator.STREAM_BYTES_ITEMS)

        # if we get an unexpected byte event, warn but ignore
        if self.traffic_model == None:
            logging.warning("No traffic model for stream bytes event")
            return True

        chanid, circid, strmid, is_outbound, bw_bytes = [int(v) for v in items[0:5]]
        ts = float(items[5])

        # TODO: secure delete
        #del items

        self.strm_bytes.setdefault(strmid, {}).setdefault(circid, [])
        self.strm_bytes[strmid][circid].append([bw_bytes, is_outbound, ts])
        return True

    STREAM_ENDED_ITEMS = 10

    # Positional event: fields is a list of Values.
    # All fields are mandatory, order matters
    # 'PRIVCOUNT_STREAM_ENDED', ChanID, CircID, StreamID, ExitPort, ReadBW, WriteBW, TimeStart, TimeEnd, RemoteHost, RemoteIP
    # See doc/TorEvents.markdown for details
    def _handle_stream_event(self, items):
        assert(len(items) == Aggregator.STREAM_ENDED_ITEMS)

        chanid, circid, strmid, port, readbw, writebw = [int(v) for v in items[0:6]]
        start, end = float(items[6]), float(items[7])
        remote_host = items[8]
        remote_ip = items[9]

        # TODO: secure delete
        #del items

        # only count streams with legitimate transfers
        totalbw = readbw + writebw
        if readbw < 0 or writebw < 0 or totalbw <= 0:
            return True

        self.circ_info.setdefault(chanid, {}).setdefault(circid, {'num_streams': {'interactive':0, 'web':0, 'p2p':0, 'other':0}, 'stream_starttimes': {'interactive':[], 'web':[], 'p2p':[], 'other':[]}})

        stream_class = Aggregator._classify_port(port)
        self.circ_info[chanid][circid]['num_streams'][stream_class] += 1
        self.circ_info[chanid][circid]['stream_starttimes'][stream_class].append(start)

        # the amount we read from the stream is bound for the client
        # the amount we write to the stream is bound to the server
        ratio = Aggregator._encode_ratio(readbw, writebw)
        lifetime = end-start

        self.secure_counters.increment('ExitStreamCount',
                                       bin=SINGLE_BIN,
                                       inc=1)
        self.secure_counters.increment('ExitStreamByteCount',
                                       bin=SINGLE_BIN,
                                       inc=totalbw)
        self.secure_counters.increment('ExitStreamOutboundByteCount',
                                       bin=writebw,
                                       inc=1)
        self.secure_counters.increment('ExitStreamInboundByteCount',
                                       bin=readbw,
                                       inc=1)
        self.secure_counters.increment('ExitStreamByteRatio',
                                       bin=ratio,
                                       inc=1)
        self.secure_counters.increment('ExitStreamLifeTime',
                                       bin=lifetime,
                                       inc=1)

        if stream_class == 'web':
            self.secure_counters.increment('ExitWebStreamCount',
                                           bin=SINGLE_BIN,
                                           inc=1)
            self.secure_counters.increment('ExitWebStreamByteCount',
                                           bin=SINGLE_BIN,
                                           inc=totalbw)
            self.secure_counters.increment('ExitWebStreamOutboundByteCount',
                                           bin=writebw,
                                           inc=1)
            self.secure_counters.increment('ExitWebStreamInboundByteCount',
                                           bin=readbw,
                                           inc=1)
            self.secure_counters.increment('ExitWebStreamByteRatio',
                                           bin=ratio,
                                           inc=1)
            self.secure_counters.increment('ExitWebStreamLifeTime',
                                           bin=lifetime,
                                           inc=1)
        elif stream_class == 'interactive':
            self.secure_counters.increment('ExitInteractiveStreamCount',
                                           bin=SINGLE_BIN,
                                           inc=1)
            self.secure_counters.increment('ExitInteractiveStreamByteCount',
                                           bin=SINGLE_BIN,
                                           inc=totalbw)
            self.secure_counters.increment('ExitInteractiveStreamOutboundByteCount',
                                           bin=writebw,
                                           inc=1)
            self.secure_counters.increment('ExitInteractiveStreamInboundByteCount',
                                           bin=readbw,
                                           inc=1)
            self.secure_counters.increment('ExitInteractiveStreamByteRatio',
                                           bin=ratio,
                                           inc=1)
            self.secure_counters.increment('ExitInteractiveStreamLifeTime',
                                           bin=lifetime,
                                           inc=1)
        elif stream_class == 'p2p':
            self.secure_counters.increment('ExitP2PStreamCount',
                                           bin=SINGLE_BIN,
                                           inc=1)
            self.secure_counters.increment('ExitP2PStreamByteCount',
                                           bin=SINGLE_BIN,
                                           inc=totalbw)
            self.secure_counters.increment('ExitP2PStreamOutboundByteCount',
                                           bin=writebw,
                                           inc=1)
            self.secure_counters.increment('ExitP2PStreamInboundByteCount',
                                           bin=readbw,
                                           inc=1)
            self.secure_counters.increment('ExitP2PStreamByteRatio',
                                           bin=ratio,
                                           inc=1)
            self.secure_counters.increment('ExitP2PStreamLifeTime',
                                           bin=lifetime,
                                           inc=1)
        elif stream_class == 'other':
            self.secure_counters.increment('ExitOtherPortStreamCount',
                                           bin=SINGLE_BIN,
                                           inc=1)
            self.secure_counters.increment('ExitOtherPortStreamByteCount',
                                           bin=SINGLE_BIN,
                                           inc=totalbw)
            self.secure_counters.increment('ExitOtherPortStreamOutboundByteCount',
                                           bin=writebw,
                                           inc=1)
            self.secure_counters.increment('ExitOtherPortStreamInboundByteCount',
                                           bin=readbw,
                                           inc=1)
            self.secure_counters.increment('ExitOtherPortStreamByteRatio',
                                           bin=ratio,
                                           inc=1)
            self.secure_counters.increment('ExitOtherPortStreamLifeTime',
                                           bin=lifetime,
                                           inc=1)

        # if we have a traffic model object, then we should use our observations to find the
        # most likely path through the HMM, and then count some aggregate statistics
        # about that path
        if self.traffic_model is not None and strmid in self.strm_bytes and circid in self.strm_bytes[strmid]:
            byte_events = self.strm_bytes[strmid][circid]
            strm_start_ts = start
            # let the model handle the model-specific counter increments
            self.traffic_model.increment_traffic_counters(strm_start_ts, byte_events, self.secure_counters)

        # clear all 'traffic' data for this stream
        # TODO: secure delete
        if strmid in self.strm_bytes:
            self.strm_bytes[strmid].pop(circid, None)
            if len(self.strm_bytes[strmid]) == 0:
                self.strm_bytes.pop(strmid, None)
        return True

    @staticmethod
    def _classify_port(port):
        '''
        Classify port into web, interactive, p2p, or other.
        '''
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

    @staticmethod
    def _encode_ratio(inval, outval):
        '''
        Calculate the log ratio between inbound and outbound traffic.
        Positive when outval > inval, and negative when inval > outval.
        Returns a non-infinite floating point value:
        - zero when inval and outval are zero,
        - a large negative number (< -100) when outval is zero, and
        - a large positive number (> 100) when inval is zero, and
        - log(base 2)(outval/inval) otherwise.
        '''
        inval = float(inval)
        outval = float(outval)
        if inval == 0.0 and outval == 0.0:
            return 0.0
        elif inval == 0.0:
            return sys.float_info.max_exp
        elif outval == 0.0:
            return sys.float_info.min_exp
        else:
            return math.log(outval/inval, 2)

    @staticmethod
    def _compute_interstream_creation_times(start_times):
        '''
        Sort start_times, and return a list of the differences between each
        pair of times.
        '''
        start_times.sort()
        isc_times = []
        for i in xrange(len(start_times)):
            if i == 0: continue
            isc_times.append(start_times[i] - start_times[i-1])
        return isc_times

    # The legacy event is still processed by the injector, but is ignored
    # by PrivCount 1.1.0 and later
    CIRCUIT_ENDED_ITEMS = 12

    def _handle_legacy_exit_circuit_event(self, fields, event_desc):
        '''
        Increment circuit-level counters and store circuit data for later
        processing. This is the legacy code that used to process the
        PRIVCOUNT_CIRCUIT_ENDED event, and now processes Exit events sent to
        it via _handle_circuit_close_event.
        '''
        event_desc = event_desc + " processing legacy circuit end event counters"

        # Extract the field values we want
        chanid = get_int_value("PreviousChannelId",
                               fields, event_desc,
                               is_mandatory=False)
        circid = get_int_value("PreviousCircuitId",
                               fields, event_desc,
                               is_mandatory=False)

        ncellsin = get_int_value("InboundExitCellCount",
                                 fields, event_desc,
                                 is_mandatory=False)
        ncellsout = get_int_value("OutboundExitCellCount",
                                  fields, event_desc,
                                  is_mandatory=False)
        readbwexit = get_int_value("InboundExitByteCount",
                                   fields, event_desc,
                                   is_mandatory=False)
        writebwexit = get_int_value("OutboundExitByteCount",
                                    fields, event_desc,
                                    is_mandatory=False)

        start = get_float_value("CreatedTimestamp",
                                fields, event_desc,
                                is_mandatory=False)
        end = get_float_value("EventTimestamp",
                              fields, event_desc,
                              is_mandatory=False)

        previp = get_ip_address_value("PreviousNodeIPAddress",
                                      fields, event_desc,
                                      is_mandatory=False,
                                      default="0.0.0.0")
        prevIsClient = get_flag_value("IsEntryFlag",
                                      fields, event_desc,
                                      is_mandatory=False,
                                      default=False)
        nextip = get_ip_address_value("NextNodeIPAddress",
                                      fields, event_desc,
                                      is_mandatory=False,
                                      default="0.0.0.0")
        nextIsEdge = get_flag_value("IsExitFlag",
                                    fields, event_desc,
                                    is_mandatory=False,
                                    default=False)

        # check they are all present
        if (chanid is None or circid is None or ncellsin is None or
            ncellsout is None or readbwexit is None or writebwexit is None or
            start is None or end is None or previp is None or
            prevIsClient is None or nextip is None or nextIsEdge is None):
            logging.warning("Unexpected missing field {}".format(event_desc))
            return False

        # Now process using the legacy code

        # we get circuit events on both exits and entries
        # stream bw info is only avail on exits
        if prevIsClient:
            # prev hop is a client, we are entry

            # only count cells ratio on active circuits with legitimate transfers
            is_active = True if ncellsin + ncellsout >= 8 else False
            if is_active:
                self.secure_counters.increment('EntryActiveCircuitCount',
                                               bin=SINGLE_BIN,
                                               inc=1)
                self.secure_counters.increment('EntryCircuitInboundCellCount',
                                               bin=ncellsin,
                                               inc=1)
                self.secure_counters.increment('EntryCircuitOutboundCellCount',
                                               bin=ncellsout,
                                               inc=1)
                self.secure_counters.increment('EntryCircuitCellRatio', bin=Aggregator._encode_ratio(ncellsin, ncellsout), inc=1)
            else:
                self.secure_counters.increment('EntryInactiveCircuitCount',
                                               bin=SINGLE_BIN,
                                               inc=1)

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
            self.cli_ips_current[previp].setdefault('num_active_completed',
                                                    0)
            self.cli_ips_current[previp].setdefault('num_inactive_completed',
                                                    0)
            if is_active:
                self.cli_ips_current[previp]['num_active_completed'] += 1
            else:
                self.cli_ips_current[previp]['num_inactive_completed'] += 1

        elif nextIsEdge:
            # prev hop is a relay and next is an edge connection, we are exit
            # don't count single-hop exits
            self.secure_counters.increment('ExitCircuitLifeTime',
                                           bin=(end - start),
                                           inc=1)

            # check if we have any stream info in this circuit
            circ_is_known, has_completed_stream = False, False
            if chanid in self.circ_info and circid in self.circ_info[chanid]:
                circ_is_known = True
                if sum(self.circ_info[chanid][circid]['num_streams'].values()) > 0:
                    has_completed_stream = True

            if circ_is_known and has_completed_stream:
                # we have circuit info and at least one stream ended on it
                self.secure_counters.increment('ExitActiveCircuitCount',
                                               bin=SINGLE_BIN,
                                               inc=1)
                self.secure_counters.increment('ExitActiveCircuitLifeTime',
                                               bin=(end - start),
                                               inc=1)

                # convenience
                counts = self.circ_info[chanid][circid]['num_streams']
                times = self.circ_info[chanid][circid]['stream_starttimes']

                # first increment general counters
                self.secure_counters.increment('ExitCircuitStreamCount',
                                               bin=sum(counts.values()),
                                               inc=1)
                for isct in Aggregator._compute_interstream_creation_times(times['web'] + times['interactive'] + times['p2p'] + times['other']):
                    self.secure_counters.increment('ExitCircuitInterStreamCreationTime',
                                                   bin=isct,
                                                   inc=1)

                # now only increment the classes that have positive counts
                if counts['web'] > 0:
                    self.secure_counters.increment('ExitWebCircuitCount',
                                                   bin=SINGLE_BIN,
                                                   inc=1)
                    self.secure_counters.increment('ExitCircuitWebStreamCount',
                                                   bin=counts['web'],
                                                   inc=1)
                    for isct in Aggregator._compute_interstream_creation_times(times['web']):
                        self.secure_counters.increment('ExitCircuitWebInterStreamCreationTime',
                                                       bin=isct,
                                                       inc=1)
                if counts['interactive'] > 0:
                    self.secure_counters.increment('ExitInteractiveCircuitCount',
                                                   bin=SINGLE_BIN,
                                                   inc=1)
                    self.secure_counters.increment('ExitCircuitInteractiveStreamCount',
                                                   bin=counts['interactive'],
                                                   inc=1)
                    for isct in Aggregator._compute_interstream_creation_times(times['interactive']):
                        self.secure_counters.increment('ExitCircuitInteractiveInterStreamCreationTime',
                                                       bin=isct,
                                                       inc=1)
                if counts['p2p'] > 0:
                    self.secure_counters.increment('ExitP2PCircuitCount',
                                                   bin=SINGLE_BIN,
                                                   inc=1)
                    self.secure_counters.increment('ExitCircuitP2PStreamCount',
                                                   bin=counts['p2p'],
                                                   inc=1)
                    for isct in Aggregator._compute_interstream_creation_times(times['p2p']):
                        self.secure_counters.increment('ExitCircuitP2PInterStreamCreationTime',
                                                       bin=isct,
                                                       inc=1)
                if counts['other'] > 0:
                    self.secure_counters.increment('ExitOtherPortCircuitCount',
                                                   bin=SINGLE_BIN,
                                                   inc=1)
                    self.secure_counters.increment('ExitCircuitOtherPortStreamCount',
                                                   bin=counts['other'],
                                                   inc=1)
                    for isct in Aggregator._compute_interstream_creation_times(times['other']):
                        self.secure_counters.increment('ExitCircuitOtherPortInterStreamCreationTime',
                                                       bin=isct,
                                                       inc=1)

            else:
                # either we dont know circ, or no streams ended on it
                self.secure_counters.increment('ExitInactiveCircuitCount',
                                               bin=SINGLE_BIN,
                                               inc=1)
                self.secure_counters.increment('ExitInactiveCircuitLifeTime',
                                               bin=(end - start),
                                               inc=1)

            # cleanup
            # TODO: secure delete
            if circ_is_known:
                # remove circ from channel
                self.circ_info[chanid].pop(circid, None)
                # if that was the last circuit on channel, remove the channel too
                if len(self.circ_info[chanid]) == 0:
                    self.circ_info.pop(chanid, None)
        return True

    CONNECTION_ENDED_ITEMS = 5

    # Positional event: fields is a list of Values.
    # All fields are mandatory, order matters
    # 'PRIVCOUNT_CONNECTION_ENDED', ChanID, TimeStart, TimeEnd, IP, isClient
    # See doc/TorEvents.markdown for details
    def _handle_connection_event(self, items):
        assert(len(items) == Aggregator.CONNECTION_ENDED_ITEMS)

        chanid = int(items[0])
        start, end = float(items[1]), float(items[2])
        ip = items[3]
        isclient = True if int(items[4]) > 0 else False

        # TODO: secure delete
        #del items

        if isclient:
            self.secure_counters.increment('EntryConnectionCount',
                                           bin=SINGLE_BIN,
                                           inc=1)
            self.secure_counters.increment('EntryConnectionLifeTime',
                                           bin=(end - start),
                                           inc=1)
        return True

    @staticmethod
    def is_hs_version_valid(fields, event_desc,
                            is_mandatory=False):
        '''
        Check that fields["HiddenServiceVersionNumber"] exists and is 2 or 3.
        See is_int_valid for details.
        '''
        return is_int_valid("HiddenServiceVersionNumber",
                            fields, event_desc,
                            is_mandatory=is_mandatory,
                            min_value=2, max_value=3)

    @staticmethod
    def get_hs_version(fields, event_desc,
                       is_mandatory=False,
                       default=None):
        '''
        Check that fields["HiddenServiceVersionNumber"] exists and is valid.
        If it is, return it as an integer.
        Otherwise, if is_mandatory is True, assert.
        Otherwise, return default.
        See is_hs_version_valid for details.
        '''
        # This should have been checked earlier
        assert Aggregator.is_hs_version_valid(fields, event_desc,
                                              is_mandatory=is_mandatory)

        return get_int_value("HiddenServiceVersionNumber",
                             fields, event_desc,
                             is_mandatory=is_mandatory,
                             default=default)

    @staticmethod
    def warn_unexpected_field_value(field_name, fields, event_desc):
        '''
        Called when we expect field_name to be a particular value, and it is
        not. Log a warning containing field_name, fields[field_name]
        (if available), and event_desc.
        '''
        if field_name in fields:
            field_value = fields[field_name]
            field_value_log = summarise_string(field_value, 20)
            value_message = "{} value '{}'".format(field_name,
                                                   field_value_log)
            full_value_message = "{} value (full value) '{}'".format(
                                                                  field_name,
                                                                  field_value)
        else:
            value_message = "missing {} value".format(field_name)
            full_value_message = value_message
        logging.warning("Unexpected {} {}. Maybe we should add a counter for it?"
                        .format(value_message, event_desc))
        logging.debug("Unexpected {} {}. Maybe we should add a counter for it?"
                      .format(full_value_message, event_desc))

    @staticmethod
    def warn_unknown_counter(counter_name, origin_desc, event_desc):
        '''
        If counter_name is an unknown counter name, log a warning containing
        origin_desc and event_desc.
        '''
        if counter_name not in get_valid_counters():
            logging.warning("Ignored unknown counter {} from {} {}. Is your PrivCount Tor version newer than your PrivCount version?"
                            .format(counter_name, origin_desc, event_desc))

    @staticmethod
    def are_circuit_id_fields_valid(fields, event_desc,
                                    is_mandatory=False,
                                    prefix=None):
        '''
        Check if the circuit id fields are valid.
        If prefix is not none, use it as the field name prefix.

        Returns True if they are all valid, False if one or more are not.
        Logs a warning using event_desc for the first field that is invalid.
        '''
        if prefix is None:
            prefix = ""

        # Channel and Circuit IDs do have maximum values (and CircuitId can't
        # be zero), but there's not much point in checking maxima: the code
        # will work regardless
        if not is_int_valid("{}ChannelId".format(prefix),
                            fields, event_desc,
                            is_mandatory=is_mandatory,
                            min_value=0):
            return False

        if not is_int_valid("{}CircuitId".format(prefix),
                            fields, event_desc,
                            is_mandatory=is_mandatory,
                            min_value=0):
            return False

        return True

    @staticmethod
    def are_circuit_common_fields_valid(fields, event_desc,
                                        is_mandatory=False,
                                        prefix=None):
        '''
        Check if the common fields across the circuit and cell events are
        valid. If is_mandatory is True, potentially mandatory fields are
        treated as mandatory. (Some fields are always optional.)
        If prefix is not none, use it as the field name prefix.

        Returns True if they are all valid, False if one or more are not.
        Logs a warning using event_desc for the first field that is invalid.
        '''
        # Validate the potentially mandatory fields

        if not is_float_valid("EventTimestamp",
                              fields, event_desc,
                              is_mandatory=is_mandatory,
                              min_value=0.0):
            return False

        # Validate the always optional fields

        # in some rare cases, NextChannelId can be missing
        if not Aggregator.are_circuit_id_fields_valid(fields, event_desc,
                                                      is_mandatory=False,
                                                      prefix="Next"):
            return False


        if not Aggregator.are_circuit_id_fields_valid(fields, event_desc,
                                                      is_mandatory=False,
                                                      prefix="Previous"):
            return False

        # We could try to validate that the position flags occur in the right
        # combinations, but the Tor patch already does that

        # These flags are only present when they are 1
        if not is_flag_valid("IsOriginFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        if not is_flag_valid("IsEntryFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        if not is_flag_valid("IsMidFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        if not is_flag_valid("IsEndFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        # Similarly, we don't bother checking subcategory combinations
        if not is_flag_valid("IsExitFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        if not is_flag_valid("IsDirFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        if not is_flag_valid("IsHSDirFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        if not is_flag_valid("IsIntroFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        if not is_flag_valid("IsRendFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        # This flag is only present for HSDir and Intro positions,
        # but is present whether it is 0 or 1
        if not is_flag_valid("IsHSClientSideFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        # This flag is only present when it is 1
        if not is_flag_valid("IsMarkedForCloseFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        # Make sure we were designed to work with the event's
        # HiddenServiceVersionNumber
        if not Aggregator.is_hs_version_valid(fields, event_desc,
                                              is_mandatory=False):
            return False

        # if everything passed, this much is ok
        return True

    @staticmethod
    def are_circuit_cell_fields_valid(fields, event_desc):
        '''
        Check if the PRIVCOUNT_CIRCUIT_CELL fields are valid.
        Returns True if they are all valid, False if one or more are not.
        Logs a warning using event_desc for the first field that is invalid.
        '''

        if not Aggregator.are_circuit_common_fields_valid(fields, event_desc,
                                                          is_mandatory=True,
                                                          prefix=None):
            return False

        # Validate the mandatory cell-specific fields


        if not is_flag_valid("IsSentFlag",
                             fields, event_desc,
                             is_mandatory=True):
            return False

        # the cell circuit id is allowed to be zero, for non-circuit cells
        if not is_int_valid("CellCircuitId",
                            fields, event_desc,
                            is_mandatory=True,
                            min_value=0):
            return False

        # 50 is an arbitrary limit, the current maximum is 14 characters
        if not is_string_valid("CellCommandString",
                               fields, event_desc,
                               is_mandatory=True,
                               min_len=1,
                               max_len=50):
            return False

        # Validate the optional cell-specific fields

        if not is_flag_valid("IsOutboundFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        if not is_int_valid("RelayCellPayloadByteCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        if not is_int_valid("RelayCellStreamId",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        # 50 is an arbitrary limit, the current maximum is 22 characters
        if not is_string_valid("RelayCellCommandString",
                               fields, event_desc,
                               is_mandatory=False,
                               min_len=1,
                               max_len=50):
            return False

        if not is_flag_valid("IsRecognizedFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        if not is_flag_valid("WasRelayCryptSuccessfulFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        # if everything passed, we're ok
        return True

    def _handle_circuit_cell_event(self, fields):
        '''
        Process a PRIVCOUNT_CIRCUIT_CELL event
        This is a tagged event: fields is a dictionary of Key=Value pairs.
        Fields may be optional, order is unimportant.

        Fields available:
        Common:
          EventTimestamp
        Cell and Circuit Common:
          EventTimestamp,
          {Previous,Next}{ChannelId,CircuitId},
          IsOriginFlag, IsEntryFlag, IsMidFlag, IsEndFlag,
          IsExitFlag, IsDirFlag, IsHSDirFlag, IsIntroFlag, IsRendFlag,
          IsHSClientSideFlag, HiddenServiceVersionNumber,
          IsMarkedForCloseFlag
        Cell-Specific:
          IsSentFlag, IsOutboundFlag,
          CellCircuitId, CellCommandString,
          RelayCellPayloadByteCount, RelayCellStreamId, RelayCellCommandString,
          IsRecognizedFlag, WasRelayCryptSuccessfulFlag
        TODO: See doc/TorEvents.markdown for all field names and definitions.

        Returns True if an event was successfully processed (or ignored).
        Never returns False: we prefer to warn about malformed events and
        continue processing.
        '''
        event_desc = "in PRIVCOUNT_CIRCUIT_CELL event"

        if not Aggregator.are_circuit_cell_fields_valid(fields, event_desc):
            # handle the event by warning (in the validator) and ignoring it
            return True

        # Extract mandatory fields

        is_sent = get_flag_value("IsSentFlag",
                                 fields, event_desc,
                                 is_mandatory=True)

        # Extract the optional fields, and give them defaults

        is_rend = get_flag_value("IsRendFlag",
                                 fields, event_desc,
                                 is_mandatory=False,
                                 default=False)

        # Extract the optional fields that don't have defaults

        # If this flag is absent, we don't know if it's client or server
        is_client = get_flag_value("IsHSClientSideFlag",
                                   fields, event_desc,
                                   is_mandatory=False)

        hs_version = Aggregator.get_hs_version(fields, event_desc,
                                               is_mandatory=False,
                                               default=None)

        # Increment counters for mandatory fields and optional fields that
        # have defaults

        # Increment counters for optional fields that don't have defaults

        # TODO: generalise, building counter names in code
        if is_client is not None and hs_version is not None:
            if is_rend and is_client and is_sent and hs_version == 2:
                self.secure_counters.increment('Rend2ClientSentCellCount',
                                               bin=SINGLE_BIN,
                                               inc=1)

        # we processed and handled the event
        return True

    @staticmethod
    def are_circuit_node_fields_valid(fields, event_desc,
                                      is_mandatory=False,
                                      prefix=None):
        '''
        Check if the circuit node fields are valid.
        If prefix is not none, use it as the field name prefix.

        Returns True if they are all valid, False if one or more are not.
        Logs a warning using event_desc for the first field that is invalid.
        '''
        if prefix is None:
            prefix = ""

        if not is_ip_address_valid("{}NodeIPAddress".format(prefix),
                                   fields, event_desc,
                                   is_mandatory=is_mandatory):
            return False

        if not is_string_valid("{}NodeFingerprint".format(prefix),
                               fields, event_desc,
                               is_mandatory=is_mandatory,
                               min_len=40,
                               max_len=40):
            return False

        # All nodes have the Running flag in their networkstatus.
        # Almost all nodes have the Valid flag as well
        # 20 is an arbitrary limit, there are currently only 10 flags
        if not is_list_valid("{}NodeRelayFlagList".format(prefix),
                               fields, event_desc,
                               is_mandatory=is_mandatory,
                               min_count=1,
                               max_count=20):
            return False

        return True

    @staticmethod
    def are_circuit_close_fields_valid(fields, event_desc):
        '''
        Check if the PRIVCOUNT_CIRCUIT_CLOSE fields are valid.
        Returns True if they are all valid, False if one or more are not.
        Logs a warning using event_desc for the first field that is invalid.
        '''

        if not Aggregator.are_circuit_common_fields_valid(fields, event_desc,
                                                          is_mandatory=True,
                                                          prefix=None):
            return False

        # Validate the mandatory circuit-specific fields
        if not is_float_valid("CreatedTimestamp",
                              fields, event_desc,
                              is_mandatory=True,
                              min_value=0.0):
            return False

        if not is_flag_valid("IsLegacyCircuitEndEventFlag",
                             fields, event_desc,
                             is_mandatory=True):
            return False

        # 50 is an arbitrary limit, the current maximum is 11 characters
        if not is_string_valid("StateString",
                               fields, event_desc,
                               is_mandatory=True,
                               min_len=1,
                               max_len=50):
            return False

        if not is_int_valid("PurposeCode",
                            fields, event_desc,
                            is_mandatory=True,
                            min_value=1,
                            max_value=20):
            return False

        # Validate the optional circuit-specific fields

        # 50 is an arbitrary limit, the current maximum is 17 characters
        if not is_string_valid("PurposeString",
                               fields, event_desc,
                               is_mandatory=False,
                               min_len=1,
                               max_len=50):
            return False

        # We don't check if the purpose and hidden service state match
        # 50 is an arbitrary limit, the current maximum is 24 characters
        if not is_string_valid("HSStateString",
                               fields, event_desc,
                               is_mandatory=False,
                               min_len=1,
                               max_len=50):
            return False

        # Check the connected node fields

        if not Aggregator.are_circuit_node_fields_valid(fields, event_desc,
                                                        is_mandatory=False,
                                                        prefix="Previous"):
            return False

        if not Aggregator.are_circuit_node_fields_valid(fields, event_desc,
                                                        is_mandatory=False,
                                                        prefix="Next"):
            return False

        # Check the related circuit fields

        if not Aggregator.are_circuit_common_fields_valid(
                                                    fields, event_desc,
                                                    is_mandatory=False,
                                                    prefix="IntroClientSink"):
            return False


        if not Aggregator.are_circuit_common_fields_valid(
                                                    fields, event_desc,
                                                    is_mandatory=False,
                                                    prefix="RendSplice"):
            return False

        # Check the cell and byte counts

        if not is_int_valid("InboundSentCellCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        if not is_int_valid("InboundReceivedCellCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        if not is_int_valid("OutboundSentCellCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        if not is_int_valid("OutboundReceivedCellCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        if not is_int_valid("InboundExitCellCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        if not is_int_valid("OutboundExitCellCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        if not is_int_valid("InboundExitByteCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        if not is_int_valid("OutboundExitByteCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        if not is_int_valid("InboundDirByteCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        if not is_int_valid("OutboundDirByteCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False

        # if everything passed, we're ok
        return True

    def _handle_circuit_close_event(self, fields):
        '''
        Process a PRIVCOUNT_CIRCUIT_CLOSE event
        This is a tagged event: fields is a dictionary of Key=Value pairs.
        Fields may be optional, order is unimportant.

        Fields available:
        Common:
          EventTimestamp
        Cell and Circuit Common:
          {Previous,Next}{ChannelId,CircuitId},
          IsOriginFlag, IsEntryFlag, IsMidFlag, IsEndFlag,
          IsExitFlag, IsDirFlag, IsHSDirFlag, IsIntroFlag, IsRendFlag,
          IsHSClientSideFlag, HiddenServiceVersionNumber,
          IsMarkedForCloseFlag
        Circuit-Specific:
          CreatedTimestamp, IsLegacyCircuitEndEventFlag,
          StateString, PurposeCode, PurposeString, HSStateString,
          {Previous,Next}Node{IPAddress,Fingerprint,RelayFlagList},
          {IntroClientSink,RendSplice}[All Cell and Circuit Common Fields],
          {Inbound,Outbound}{Sent,Received}CellCount,
          {Inbound,Outbound}Exit{Cell,Byte}Count,
          {Inbound,Outbound}DirByteCount
        TODO: See doc/TorEvents.markdown for all field names and definitions.

        Calls _handle_legacy_exit_circuit_event to increment legacy counters
        that were based on the legacy PRIVCOUNT_CIRCUIT_ENDED event.

        Returns True if an event was successfully processed (or ignored).
        Never returns False: we prefer to warn about malformed events and
        continue processing.
        '''
        event_desc = "in PRIVCOUNT_CIRCUIT_CLOSE event"

        if not Aggregator.are_circuit_close_fields_valid(fields, event_desc):
            # handle the event by warning (in the validator) and ignoring it
            return True

        # handle the legacy event
        is_legacy = get_flag_value("IsLegacyCircuitEndEventFlag",
                                   fields, event_desc,
                                   is_mandatory=True)

        if is_legacy:
            if not self._handle_legacy_exit_circuit_event(fields, event_desc):
                logging.warning("Error while processing legacy circuit event with '{}' {}"
                                .format(" ".join(sorted(fields)), event_desc))

        # Extract mandatory fields

        # Extract the optional fields, and give them defaults

        # Unused, included for completeness
        is_origin = get_flag_value("IsOriginFlag",
                                   fields, event_desc,
                                   is_mandatory=False,
                                   default=False)

        is_entry = get_flag_value("IsEntryFlag",
                                  fields, event_desc,
                                  is_mandatory=False,
                                  default=False)

        is_mid = get_flag_value("IsMidFlag",
                                fields, event_desc,
                                is_mandatory=False,
                                default=False)

        is_end = get_flag_value("IsEndFlag",
                                fields, event_desc,
                                is_mandatory=False,
                                default=False)

        is_single_hop = is_entry and is_end

        # The end position can be exit, dir, hsdir, intro, or rend
        # (intro circuits can also be middle circuits)
        # Exactly one of these should be true, except for unused preemptive
        # circuits
        is_exit = get_flag_value("IsExitFlag",
                                 fields, event_desc,
                                 is_mandatory=False,
                                 default=False)

        # Unused, included for completeness
        is_dir = get_flag_value("IsDirFlag",
                                 fields, event_desc,
                                 is_mandatory=False,
                                 default=False)

        # Unused, included for completeness
        is_hsdir = get_flag_value("IsHSDirFlag",
                                 fields, event_desc,
                                 is_mandatory=False,
                                 default=False)

        # Unused, included for completeness
        is_intro = get_flag_value("IsIntroFlag",
                                 fields, event_desc,
                                 is_mandatory=False,
                                 default=False)

        is_rend = get_flag_value("IsRendFlag",
                                 fields, event_desc,
                                 is_mandatory=False,
                                 default=False)

        # Extract the optional fields that don't have defaults

        # If this flag is absent, we don't know if it's client or server
        is_client = get_flag_value("IsHSClientSideFlag",
                                   fields, event_desc,
                                   is_mandatory=False,
                                   default=None)

        hs_version = Aggregator.get_hs_version(fields, event_desc,
                                               is_mandatory=False,
                                               default=None)

        # Increment counters for mandatory fields and optional fields that
        # have defaults

        # Increment counters for optional fields that don't have defaults

        # TODO: generalise, building counter names in code

        # Positions: at least one of these flags is true for each circuit

        # Unused, included for completeness
        if is_origin:
            self.secure_counters.increment('OriginCircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)

        if is_entry:
            self.secure_counters.increment('EntryCircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)

        if is_mid:
            self.secure_counters.increment('MidCircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)

        if is_end:
            self.secure_counters.increment('EndCircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)

        # Unused, included for completeness
        if is_single_hop:
            self.secure_counters.increment('SingleHopCircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)

        # End subcategories
        if is_exit:
            # includes single-hop exits (which aren't supposed to exist)
            self.secure_counters.increment('ExitCircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)
            # Combined counters: collected so there is only one lot of noise added
            # ExitAndRend2ClientCircuitCount = ExitCircuitCount + Rend2ClientCircuitCount
            self.secure_counters.increment('ExitAndRend2ClientCircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)
            # ExitAndRend2ServiceCircuitCount = ExitCircuitCount + Rend2ServiceCircuitCount
            self.secure_counters.increment('ExitAndRend2ServiceCircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)

        # Unused, included for completeness
        if is_dir:
            self.secure_counters.increment('DirCircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)

        # Unused, included for completeness
        if is_hsdir and hs_version is not None and hs_version == 2:
            self.secure_counters.increment('HSDir2CircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)

        # Unused, included for completeness
        if is_intro and hs_version is not None and hs_version == 2:
            self.secure_counters.increment('Intro2CircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)

        if is_rend and hs_version is not None and hs_version == 2:
            self.secure_counters.increment('Rend2CircuitCount',
                                           bin=SINGLE_BIN,
                                           inc=1)
            # ignore circuits where we don't know the client/server flag
            if is_client is not None:
                if is_client:
                    # Unused, included for completeness
                    self.secure_counters.increment('Rend2ClientCircuitCount',
                                                   bin=SINGLE_BIN,
                                                   inc=1)
                    # Combined counters: collected so there is only one lot of noise added
                    # ExitAndRend2ClientCircuitCount = ExitCircuitCount + Rend2ClientCircuitCount
                    # Use double quotes, so test_counter_match.sh doesn't think it's a duplicate
                    self.secure_counters.increment("ExitAndRend2ClientCircuitCount",
                                                   bin=SINGLE_BIN,
                                                   inc=1)

                    if is_single_hop:
                        # Unused, included for completeness
                        self.secure_counters.increment(
                                          'Rend2Tor2WebClientCircuitCount',
                                          bin=SINGLE_BIN,
                                          inc=1)
                    else:
                        self.secure_counters.increment(
                                          'Rend2MultiHopClientCircuitCount',
                                          bin=SINGLE_BIN,
                                          inc=1)
                else:
                    self.secure_counters.increment('Rend2ServiceCircuitCount',
                                                   bin=SINGLE_BIN,
                                                   inc=1)
                    # Combined counters: collected so there is only one lot of noise added
                    # ExitAndRend2ServiceCircuitCount = ExitCircuitCount + Rend2ServiceCircuitCount
                    # Use double quotes, so test_counter_match.sh doesn't think it's a duplicate
                    self.secure_counters.increment("ExitAndRend2ServiceCircuitCount",
                                                   bin=SINGLE_BIN,
                                                   inc=1)
                    if is_single_hop:
                        self.secure_counters.increment(
                                          'Rend2SingleOnionServiceCircuitCount',
                                          bin=SINGLE_BIN,
                                          inc=1)
                    else:
                        # Unused, included for completeness
                        self.secure_counters.increment(
                                          'Rend2MultiHopServiceCircuitCount',
                                          bin=SINGLE_BIN,
                                          inc=1)

        # we processed and handled the event
        return True

    @staticmethod
    def is_allowed_version_valid(field_name, fields, event_desc,
                                 allowed_version=None):
        '''
        If allowed_version is not None, check if fields[field_name] exists,
        and if allowed_version is the same as the hidden service version.
        If it is not, return False and log a warning using event_desc.

        Otherwise, return True (the event should be processed).
        '''
        if allowed_version is None:
            # No version check
            return True

        assert allowed_version == 2 or allowed_version == 3

        hs_version = Aggregator.get_hs_version(fields, event_desc,
                                               is_mandatory=True)

        # this should have been checked earlier
        assert hs_version == 2 or hs_version == 3

        if hs_version != allowed_version and field_name in fields:
            logging.warning("Ignored unexpected v{} {} {}"
                            .format(hs_version, field_name, event_desc))
            return False

        return True

    @staticmethod
    def is_descriptor_byte_count_valid(field_name, fields, event_desc):
        '''
        If fields[field_name] exists, checks that it is below the expected
        maximum descriptor byte count for the hidden service version.
        (There is no minimum.)
        See is_int_valid for details.
        '''
        hs_version = Aggregator.get_hs_version(fields, event_desc,
                                               is_mandatory=True)
        if hs_version is None:
            return False

        assert hs_version == 2 or hs_version == 3
        # The hard-coded v2 limit is 20kB, but services could exceed that
        # So let's start warning at twice that.
        # The default v3 limit is 50kB, but can be changed in the consensus
        # So let's start warning if the actual v3 byte count is > 1MB
        desc_max = 2*20*1024 if hs_version == 2 else 1024*1024

        return is_int_valid(field_name,
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0,
                            max_value=desc_max)

    @staticmethod
    def are_hsdir_common_fields_valid(fields, event_desc):
        '''
        Check if the common fields across HSDir events are valid.
        Returns True if they are all valid, False if one or more are not.
        Logs a warning using event_desc for the first field that is invalid.
        '''
        # Validate the mandatory fields

        # Make sure we were designed to work with the event's
        # HiddenServiceVersionNumber
        if not Aggregator.is_hs_version_valid(fields, event_desc,
                                              is_mandatory=True):
            return False

        hs_version = Aggregator.get_hs_version(fields, event_desc,
                                               is_mandatory=True)

        # Check the event timestamp
        if not is_float_valid("EventTimestamp",
                              fields, event_desc,
                              is_mandatory=True,
                              min_value=0.0):
            return False

        # Validate the mandatory fields

        # Check the cache flags

        if not is_flag_valid("HasExistingCacheEntryFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False

        # Check the Intro Counts and Descriptor Byte Count

        if not Aggregator.is_descriptor_byte_count_valid(
                                                "EncodedIntroPointByteCount",
                                                fields, event_desc):
            return False

        if not Aggregator.is_descriptor_byte_count_valid(
                                                "EncodedDescriptorByteCount",
                                                fields, event_desc):
            return False

        intro_bytes = get_int_value("EncodedIntroPointByteCount",
                                    fields, event_desc,
                                    is_mandatory=False)
        desc_bytes = get_int_value("EncodedDescriptorByteCount",
                                   fields, event_desc,
                                   is_mandatory=False)
        if intro_bytes is not None and desc_bytes is not None:
            # it is ok if both are zero
            if intro_bytes > desc_bytes:
                logging.warning("Ignored EncodedIntroPointByteCount {} greater than EncodedDescriptorByteCount {} {}"
                              .format(intro_bytes, desc_bytes, event_desc))
                return False

        # Check the v2 fields

        if not is_int_valid("IntroPointCount",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0,
                            max_value=10):
            return False

        # Version 3 always has intro points encrypted
        if not Aggregator.is_allowed_version_valid("IntroPointCount",
                                                   fields, event_desc,
                                                   allowed_version=2):
            return False

        intro_count = get_int_value("IntroPointCount",
                                    fields, event_desc,
                                    is_mandatory=False)

        # Version 3 always has intro points encrypted, so we can't tell if it
        # uses client auth
        if not is_flag_valid("RequiresClientAuthFlag",
                             fields, event_desc,
                             is_mandatory=False):
            return False
        if not Aggregator.is_allowed_version_valid("RequiresClientAuthFlag",
                                                   fields, event_desc,
                                                   allowed_version=2):
            return False


        if not is_float_valid("DescriptorCreationTime",
                              fields, event_desc,
                              is_mandatory=False,
                              min_value=0.0):
            return False
        if not Aggregator.is_allowed_version_valid("DescriptorCreationTime",
                                                   fields, event_desc,
                                                   allowed_version=2):
            return False

        # Check the v3 fields

        # Version 2 doesn't have this field
        if not is_int_valid("RevisionNumber",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=0):
            return False
        if not Aggregator.is_allowed_version_valid("RevisionNumber",
                                                   fields, event_desc,
                                                   allowed_version=3):
            return False

        # We don't collect these fields, because all services put the same
        # value in every descriptor. But if they are not, log a warning, and
        # keep processing the event.

        # Is v2 SupportedProtocolBitfield 0xc (1 << 2 + 1 << 3)?
        const_bitfield_str = '0xc'
        const_bitfield_strlen = len(const_bitfield_str)
        if not is_string_valid("SupportedProtocolBitfield",
                               fields, event_desc,
                               is_mandatory=False,
                               min_len=const_bitfield_strlen,
                               max_len=const_bitfield_strlen):
            Aggregator.warn_unexpected_field_value("SupportedProtocolBitfield",
                                                   fields, event_desc)
        bitfield_str = get_string_value("SupportedProtocolBitfield",
                                        fields, event_desc,
                                        is_mandatory=False)
        if bitfield_str is not None and bitfield_str != const_bitfield_str:
            Aggregator.warn_unexpected_field_value("SupportedProtocolBitfield",
                                                   fields, event_desc)
        if not Aggregator.is_allowed_version_valid("SupportedProtocolBitfield",
                                                   fields, event_desc,
                                                   allowed_version=2):
            return False

        # Is the number of fingerprints the same as the intro point count?
        if not is_list_valid("IntroPointFingerprintList",
                             fields, event_desc,
                             is_mandatory=False,
                             min_count=0,
                             max_count=10):
            Aggregator.warn_unexpected_field_value("IntroPointFingerprintList",
                                                   fields, event_desc)

        # Version 3 always has intro points encrypted
        if not Aggregator.is_allowed_version_valid("IntroPointFingerprintList",
                                                   fields, event_desc,
                                                   allowed_version=2):
            return False

        # That's not quite enough, we need to check the values are equal
        intro_list = get_list_value("IntroPointFingerprintList",
                                    fields, event_desc,
                                    is_mandatory=False)
        if intro_count is None and intro_list is None:
            # that's ok
            pass
        elif intro_count is None or intro_list is None:
            # Both should be None, or neither should be None
            Aggregator.warn_unexpected_field_value("IntroPointFingerprintList",
                                                   fields, event_desc)
        elif intro_count != len(intro_list):
            # Count mismatch
            Aggregator.warn_unexpected_field_value("IntroPointFingerprintList",
                                                   fields, event_desc)

        # Is v3 DescriptorLifetime 3 hours?
        const_lifetime = 3*60*60
        if not is_int_valid("DescriptorLifetime",
                            fields, event_desc,
                            is_mandatory=False,
                            min_value=const_lifetime,
                            max_value=const_lifetime):
            Aggregator.warn_unexpected_field_value("DescriptorLifetime",
                                                   fields, event_desc)
        if not Aggregator.is_allowed_version_valid("DescriptorLifetime",
                                                   fields, event_desc,
                                                   allowed_version=3):
            return False

        # if everything passed, this much is ok
        return True

    @staticmethod
    def are_hsdir_stored_fields_valid(fields, event_desc):
        '''
        Check if the PRIVCOUNT_HSDIR_CACHE_STORE fields are valid.
        Returns True if they are all valid, False if one or more are not.
        Logs a warning using event_desc for the first field that is invalid.
        '''

        if not Aggregator.are_hsdir_common_fields_valid(fields, event_desc):
            return False

        hs_version = Aggregator.get_hs_version(fields, event_desc,
                                               is_mandatory=True)

        # Validate the mandatory cache fields

        # 50 is an arbitrary limit, the current maximum is 11 characters
        if not is_string_valid("CacheReasonString",
                               fields, event_desc,
                               is_mandatory=True,
                               min_len=1,
                               max_len=50):
            return False

        reason_str = get_string_value("CacheReasonString",
                                      fields, event_desc,
                                      is_mandatory=True)

        if not is_flag_valid("WasAddedToCacheFlag",
                             fields, event_desc,
                             is_mandatory=True):
            return False

        # Validate the optional cache fields

        has_existing = get_flag_value("HasExistingCacheEntryFlag",
                                      fields, event_desc,
                                      is_mandatory=False)

        # Some CacheReasonStrings must have HasExistingCacheEntryFlag
        if reason_str == "Expired" or reason_str == "Future":
            assert hs_version == 2 or hs_version == 3
            if hs_version != 2:
                logging.warning("Ignored CacheReasonString {} with HiddenServiceVersionNumber {}, HiddenServiceVersionNumber must be 2 {}"
                              .format(reason_str, hs_version, event_desc))
                return False
            if has_existing is None:
                logging.warning("Ignored CacheReasonString {} with HasExistingCacheEntryFlag None, HasExistingCacheEntryFlag must be 1 or 0 {}"
                                .format(reason_str, event_desc))
                return False

        # if everything passed, we're ok
        return True

    def _increment_hsdir_stored_counters(self, counter_suffix, hs_version,
                                         reason_str, was_added, has_existing,
                                         has_client_auth,
                                         event_desc,
                                         bin=SINGLE_BIN,
                                         inc=1,
                                         log_missing_counters=True):
        '''
        Increment bin by inc for the set of counters ending in
        counter_suffix, using hs_version, reason_str, was_added,
        has_existing, and has_client_auth to create the counter names.
        If log_missing_counters, warn the operator when a requested counter
        is not in the table in counters.py. Otherwise, unknown names are
        ignored.
        '''
        # create counter names from version, reason_str and was_added
        reason_str = reason_str.title()
        added_str = "Add" if was_added else "Reject"

        store_counter = "HSDir{}Store{}".format(hs_version,
                                                counter_suffix)
        added_counter = "HSDir{}Store{}{}".format(hs_version,
                                                  added_str,
                                                  counter_suffix)
        if has_client_auth is not None:
            # v2 only: we checked in are_hsdir_stored_fields_valid()
            assert hs_version == 2
            auth_str = "ClientAuth" if has_client_auth else "NoClientAuth"
            auth_counter = "HSDir{}Store{}{}".format(hs_version,
                                                     auth_str,
                                                     counter_suffix)
            added_auth_counter = "HSDir{}Store{}{}{}".format(hs_version,
                                                             added_str,
                                                             auth_str,
                                                             counter_suffix)
        # based on added_counter
        if reason_str == "Expired" or reason_str == "Future":
            # v2 only: we checked in are_hsdir_stored_fields_valid()
            assert hs_version == 2
            assert has_existing is not None
            assert has_client_auth is not None
            existing_str = "HaveCached" if has_existing else "NoCached"
            action_counter = "HSDir{}Store{}{}{}{}".format(hs_version,
                                                           added_str,
                                                           reason_str,
                                                           existing_str,
                                                           counter_suffix)
            action_auth_counter = "HSDir{}Store{}{}{}{}{}".format(
                                                                hs_version,
                                                                added_str,
                                                                reason_str,
                                                                existing_str,
                                                                auth_str,
                                                                counter_suffix)
        else:
            # The action already tells us whether there was an existing
            # descriptor. See doc/CounterDefinitions.markdown for details.
            action_counter = "HSDir{}Store{}{}{}".format(hs_version,
                                                         added_str,
                                                         reason_str,
                                                         counter_suffix)
            if has_client_auth is not None:
                # v2 only: we checked in are_hsdir_stored_fields_valid()
                assert hs_version == 2
                assert has_client_auth is not None
                action_auth_counter = "HSDir{}Store{}{}{}{}".format(
                                                                hs_version,
                                                                added_str,
                                                                reason_str,
                                                                auth_str,
                                                                counter_suffix)

        # warn the operator if we don't know the counter name
        if log_missing_counters:
            Aggregator.warn_unknown_counter(store_counter,
                                            counter_suffix,
                                            event_desc)
            added_origin = "WasAddedToCacheFlag and {}".format(counter_suffix)
            Aggregator.warn_unknown_counter(added_counter,
                                            added_origin,
                                            event_desc)
            action_origin = "CacheReasonString and HasExistingCacheEntryFlag and {}".format(counter_suffix)
            Aggregator.warn_unknown_counter(action_counter,
                                            action_origin,
                                            event_desc)

            if has_client_auth is not None:
                # v2 only: we checked in are_hsdir_stored_fields_valid()
                assert hs_version == 2

                auth_origin = "RequiresClientAuthFlag and {}".format(counter_suffix)
                Aggregator.warn_unknown_counter(auth_counter,
                                                auth_origin,
                                                event_desc)
                added_auth_origin = "WasAddedToCacheFlag and RequiresClientAuthFlag and {}".format(counter_suffix)
                Aggregator.warn_unknown_counter(added_auth_counter,
                                                added_auth_origin,
                                                event_desc)
                action_auth_origin = "CacheReasonString and HasExistingCacheEntryFlag and RequiresClientAuthFlag and {}".format(counter_suffix)
                Aggregator.warn_unknown_counter(action_auth_counter,
                                                action_auth_origin,
                                                event_desc)

        # Increment the counters
        self.secure_counters.increment(store_counter,
                                       bin=bin,
                                       inc=inc)
        self.secure_counters.increment(added_counter,
                                       bin=bin,
                                       inc=inc)
        self.secure_counters.increment(action_counter,
                                       bin=bin,
                                       inc=inc)
        if has_client_auth is not None:
            # v2 only: we checked in are_hsdir_stored_fields_valid()
            assert hs_version == 2

            self.secure_counters.increment(auth_counter,
                                           bin=bin,
                                           inc=inc)
            self.secure_counters.increment(added_auth_counter,
                                           bin=bin,
                                           inc=inc)
            self.secure_counters.increment(action_auth_counter,
                                           bin=bin,
                                           inc=inc)

    def _handle_hsdir_stored_event(self, fields):
        '''
        Process a PRIVCOUNT_HSDIR_CACHE_STORE event
        This is a tagged event: fields is a dictionary of Key=Value pairs.
        Fields may be optional, order is unimportant.

        Fields used:
        Common:
          HiddenServiceVersionNumber, EventTimestamp,
          CacheReasonString, HasExistingCacheEntryFlag, WasAddedToCacheFlag,
          EncodedDescriptorByteCount, EncodedIntroPointByteCount
        v2:
          DescriptorCreationTime, SupportedProtocolBitfield,
          RequiresClientAuthFlag, IntroPointCount, IntroPointFingerprintList
        v3:
          RevisionNumber, DescriptorLifetime

        See doc/TorEvents.markdown for all field names and definitions.
        Returns True if an event was successfully processed (or ignored).
        Never returns False: we prefer to warn about malformed events and
        continue processing.
        '''
        event_desc = "in PRIVCOUNT_HSDIR_CACHE_STORE event"

        if not Aggregator.are_hsdir_stored_fields_valid(fields, event_desc):
            # handle the event by warning (in the validator) and ignoring it
            return True

        # Extract mandatory fields
        hs_version = Aggregator.get_hs_version(fields, event_desc,
                                               is_mandatory=True)
        event_ts = get_float_value("EventTimestamp",
                                   fields, event_desc,
                                   is_mandatory=True)
        reason_str = get_string_value("CacheReasonString",
                                      fields, event_desc,
                                      is_mandatory=True)
        was_added = get_flag_value("WasAddedToCacheFlag",
                                   fields, event_desc,
                                   is_mandatory=True)

        # Extract the optional fields
        # Cache common
        has_existing = get_flag_value("HasExistingCacheEntryFlag",
                                      fields, event_desc,
                                      is_mandatory=False)
        # Intro / Descriptor common
        intro_bytes = get_int_value("EncodedIntroPointByteCount",
                                    fields, event_desc,
                                    is_mandatory=False)
        desc_bytes = get_int_value("EncodedDescriptorByteCount",
                                   fields, event_desc,
                                   is_mandatory=False)
        create_time = get_float_value("DescriptorCreationTime",
                                      fields, event_desc,
                                      is_mandatory=False)
        # Intro / Descriptor v2
        has_client_auth = get_flag_value("RequiresClientAuthFlag",
                                         fields, event_desc,
                                         is_mandatory=False)
        intro_count = get_int_value("IntroPointCount",
                                    fields, event_desc,
                                    is_mandatory=False)

        # Descriptor v3
        revision_num = get_int_value("RevisionNumber",
                                     fields, event_desc,
                                     is_mandatory=False)

        # Increment counters for mandatory fields
        # These are the base counters that cover all the upload cases
        self._increment_hsdir_stored_counters("Count",
                                              hs_version,
                                              reason_str,
                                              was_added,
                                              has_existing,
                                              has_client_auth,
                                              event_desc,
                                              bin=SINGLE_BIN,
                                              inc=1,
                                              log_missing_counters=True)

        # Increment counters for common optional fields

        if intro_bytes is not None:
            self._increment_hsdir_stored_counters("IntroByteCount",
                                                  hs_version,
                                                  reason_str,
                                                  was_added,
                                                  has_existing,
                                                  has_client_auth,
                                                  event_desc,
                                                  bin=SINGLE_BIN,
                                                  inc=intro_bytes,
                                                  log_missing_counters=False)
            self._increment_hsdir_stored_counters("IntroByteHistogram",
                                                  hs_version,
                                                  reason_str,
                                                  was_added,
                                                  has_existing,
                                                  has_client_auth,
                                                  event_desc,
                                                  bin=intro_bytes,
                                                  inc=1,
                                                  log_missing_counters=False)
        if desc_bytes is not None:
            self._increment_hsdir_stored_counters("DescriptorByteCount",
                                                  hs_version,
                                                  reason_str,
                                                  was_added,
                                                  has_existing,
                                                  has_client_auth,
                                                  event_desc,
                                                  bin=SINGLE_BIN,
                                                  inc=desc_bytes,
                                                  log_missing_counters=False)
            self._increment_hsdir_stored_counters("DescriptorByteHistogram",
                                                  hs_version,
                                                  reason_str,
                                                  was_added,
                                                  has_existing,
                                                  has_client_auth,
                                                  event_desc,
                                                  bin=desc_bytes,
                                                  inc=1,
                                                  log_missing_counters=False)

        # Increment counters for v2 optional fields

        if intro_count is not None:
            # we checked in are_hsdir_stored_fields_valid()
            assert hs_version == 2
            # we don't bother collecting detailed rejection subcategories
            # to add rejection counters, their names to the list in counter.py
            self._increment_hsdir_stored_counters("IntroPointHistogram",
                                                  hs_version,
                                                  reason_str,
                                                  was_added,
                                                  has_existing,
                                                  has_client_auth,
                                                  event_desc,
                                                  bin=intro_count,
                                                  inc=1,
                                                  log_missing_counters=False)
        if create_time is not None:
            # we checked in are_hsdir_stored_fields_valid()
            assert hs_version == 2
            # create_time is truncated to the nearest hour
            delay_time = event_ts - create_time
            self._increment_hsdir_stored_counters("UploadDelayTime",
                                                  hs_version,
                                                  reason_str,
                                                  was_added,
                                                  has_existing,
                                                  has_client_auth,
                                                  event_desc,
                                                  bin=delay_time,
                                                  inc=1,
                                                  log_missing_counters=False)

        # Increment counters for v3 optional fields

        if revision_num is not None:
            # we checked in are_hsdir_stored_fields_valid()
            assert hs_version == 3
            self._increment_hsdir_stored_counters("RevisionHistogram",
                                                  hs_version,
                                                  reason_str,
                                                  was_added,
                                                  has_existing,
                                                  has_client_auth,
                                                  event_desc,
                                                  bin=revision_num,
                                                  inc=1,
                                                  log_missing_counters=False)
        # we processed and handled the event
        return True

    def _do_rotate(self):
        '''
        This function is called using LoopingCall, so any exceptions will be
        turned into log messages.
        '''
        logging.info("rotating circuit window now, {}".format(format_last_event_time_since(self.last_event_time)))

        # it is safe to count the first rotation, because Tor only sends us
        # events that started inside the collection period
        client_ips_active = 0
        client_ips_inactive = 0

        # cli_ips_previous are the IPs from 2*period to period seconds ago,
        # or are empty for the first rotation
        for ip in self.cli_ips_previous:
            client = self.cli_ips_previous[ip]

            if client['is_active']:
                client_ips_active += 1
            else:
                client_ips_inactive += 1

            num_active_completed = client.get('num_active_completed', 0)
            self.secure_counters.increment('EntryClientIPActiveCircuitCount',
                                           bin=num_active_completed,
                                           inc=1)
            num_inactive_completed = client.get('num_inactive_completed', 0)
            self.secure_counters.increment('EntryClientIPInactiveCircuitCount',
                                           bin=num_inactive_completed,
                                           inc=1)

        self.secure_counters.increment('EntryClientIPCount',
                                       bin=SINGLE_BIN,
                                       inc=(client_ips_active + client_ips_inactive))
        self.secure_counters.increment('EntryActiveClientIPCount',
                                       bin=SINGLE_BIN,
                                       inc=client_ips_active)
        self.secure_counters.increment('EntryInactiveClientIPCount',
                                       bin=SINGLE_BIN,
                                       inc=client_ips_inactive)

        # reset for next interval
        # make cli_ips_previous the IPs from period to 0 seconds ago
        # TODO: secure delete IP addresses
        self.cli_ips_previous = self.cli_ips_current
        self.cli_ips_current = {}
        self.cli_ips_rotated = time()
        self.num_rotations += 1
