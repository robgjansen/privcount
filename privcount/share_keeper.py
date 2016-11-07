'''
Created on Dec 12, 2015

@author: rob
'''
import os
import logging
import cPickle as pickle

from copy import deepcopy

from twisted.internet import reactor, ssl
from twisted.internet.protocol import ReconnectingClientFactory

from protocol import PrivCountClientProtocol
from tally_server import log_tally_server_status
from util import SecureCounters, log_error, get_public_digest, generate_keypair, get_serialized_public_key, load_private_key_file, decrypt, normalise_path, counter_modulus, add_counter_limits_to_config, check_noise_weight_config

import yaml

class ShareKeeper(ReconnectingClientFactory):
    '''
    receive key share data from the DC message receiver
    keep the shares during collection epoch
    send the shares to the TS at end of epoch
    '''

    def __init__(self, config_filepath):
        self.config_filepath = normalise_path(config_filepath)
        self.config = None
        self.start_config = None
        self.keystore = None

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
                self.keystore = state['keystore']

    def stopFactory(self):
        # TODO
        return
        state_filepath = normalise_path(self.config['state'])
        if self.keystore is not None:
            # export everything that would be needed to survive an app restart
            state = {'keystore': self.keystore}
            with open(state_filepath, 'w') as fout:
                pickle.dump(state, fout)

    def run(self):
        # load initial config
        self.refresh_config()
        if self.config is None:
            logging.critical("cannot start due to error in config file")
            return

        logging.info("running share keeper using RSA public key id '{}'".format(self.config['name']))

        # connect to the tally server, register, and wait for commands
        self.do_checkin()
        reactor.run() # pylint: disable=E1101

    def get_status(self): # called by protocol
        return {'type':'ShareKeeper', 'name':self.config['name'],
        'state': 'active' if self.keystore is not None else 'idle', 'public_key':get_serialized_public_key(self.config['key'])}

    def set_server_status(self, status): # called by protocol
        log_tally_server_status(status)

    def do_checkin(self): # called by protocol
        # turn on reconnecting mode and reset backoff
        self.resetDelay()
        self.refresh_config()
        ts_ip = self.config['tally_server_info']['ip']
        ts_port = self.config['tally_server_info']['port']
        logging.info("checking in with TallyServer at {}:{}".format(ts_ip, ts_port))
        reactor.connectSSL(ts_ip, ts_port, self, ssl.ClientContextFactory()) # pylint: disable=E1101

    def do_start(self, config): # called by protocol
        '''
        this is called when we receive a command from the TS to start a new
        collection phase
        return None if failure, otherwise the protocol will encode the result
        in json and send it back to TS
        '''
        logging.info("got command to start new collection phase")
        # keep the start config to send to the TS at the end of the collection
        # deepcopy so we can delete the (encrypted) secrets from the shares
        self.start_config = deepcopy(config)
        # discard the secrets
        # we haven't checked if any shares are present, so don't assume
        for share in self.start_config.get('shares', []):
            # this is still encrypted, so there's no need for a secure delete
            del share['secret']
            share['secret'] = "(encrypted blinding share, deleted by share keeper)"

        if ('shares' not in config or 'counters' not in config or
            'noise_weight' not in config or 'dc_threshold' not in config):
            logging.warning("start command from tally server cannot be completed due to missing data")
            return None

        # if the noise weights don't pass the validity checks, fail
        if not check_noise_weight_config(config['noise_weight'],
                                         config['dc_threshold']):
            return None

        self.keystore = SecureCounters(config['counters'], counter_modulus())
        share_list = config['shares']

        private_key = load_private_key_file(self.config['key'])
        for share in share_list:
            encrypted_secret = share['secret']
            secret = decrypt(private_key, encrypted_secret)
            # TODO: secure delete
            share['secret'] = secret
            blinding_result = self.keystore.import_blinding_share(share)
            if not blinding_result:
                # the structure of the imported share did not match the
                # configured counters
                # this is likely a configuration error or a programming bug,
                # but there is also no way to detect the TS modifying the data
                logging.warning("failed to import blinding share {} config {}",
                                share, config)
                # TODO: secure delete
                del private_key
                return None

        logging.info("successfully started and imported {} blinding shares for {} counters".format(len(share_list), len(config['counters'])))
        # TODO: secure delete
        del private_key
        return {}

    def do_stop(self, config): # called by protocol
        '''
        the TS wants us to stop the current collection phase
        they may or may not want us to send back our counters
        return None if failure, otherwise json will encode result back to the TS
        '''
        logging.info("got command to stop collection phase")
        if 'send_counters' not in config:
            return None

        wants_counters = 'send_counters' in config and config['send_counters'] is True
        logging.info("tally server {} final counts".format("wants" if wants_counters else "does not want"))

        response_counts = None
        if wants_counters:
            # send our counts, its an error if we dont have any
            if self.keystore is not None:
                response_counts = self.keystore.detach_counts()
                logging.info("sending counts from {} counters".format(len(response_counts)))
        else:
            # this is never an error, but they dont want anything
            response_counts = {}

        del self.keystore
        self.keystore = None
        logging.info("collection phase was stopped")
        response = {}
        response['Counts'] = response_counts
        # even though the counter limits are hard-coded, include them anyway
        response['Config'] = add_counter_limits_to_config(self.config)
        # and include the config sent by the tally server in do_start
        if self.start_config is not None:
            response['Config']['Start'] = self.start_config
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
            sk_conf = conf['share_keeper']

            # if key path is not specified, look at default path, or generate a new key
            if 'key' in sk_conf:
                sk_conf['key'] = normalise_path(sk_conf['key'])
                assert os.path.exists(sk_conf['key'])
            else:
                sk_conf['key'] = normalise_path('privcount.rsa_key.pem')
                if not os.path.exists(sk_conf['key']):
                    generate_keypair(sk_conf['key'])

            sk_conf['name'] = get_public_digest(sk_conf['key'])

            sk_conf['state'] = normalise_path(sk_conf['state'])
            assert os.path.exists(os.path.dirname(sk_conf['state']))

            assert sk_conf['tally_server_info']['ip'] is not None
            assert sk_conf['tally_server_info']['port'] > 0

            if self.config == None:
                self.config = sk_conf
                logging.info("using config = %s", str(self.config))
            else:
                changed = False
                for k in sk_conf:
                    if k not in self.config or sk_conf[k] != self.config[k]:
                        logging.info("updated config for key {} from {} to {}".format(k, self.config[k], sk_conf[k]))
                        self.config[k] = sk_conf[k]
                        changed = True
                if not changed:
                    logging.debug('no config changes found')

        except AssertionError:
            logging.warning("problem reading config file: invalid data")
            log_error()
        except KeyError:
            logging.warning("problem reading config file: missing required keys")
            log_error()
