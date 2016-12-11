'''
Created on Dec 12, 2015

@author: rob
'''
import os
import logging
import cPickle as pickle
import yaml

from copy import deepcopy

from twisted.internet import reactor, ssl
from twisted.internet.protocol import ReconnectingClientFactory

from privcount.counter import SecureCounters, counter_modulus, add_counter_limits_to_config, combine_counters
from privcount.crypto import get_public_digest, generate_keypair, get_serialized_public_key, load_private_key_file, decrypt
from privcount.log import log_error
from privcount.protocol import PrivCountClientProtocol
from privcount.node import PrivCountClient
from privcount.util import normalise_path, choose_secret_handshake_path

class ShareKeeper(ReconnectingClientFactory, PrivCountClient):
    '''
    receive key share data from the DC message receiver
    keep the shares during collection epoch
    send the shares to the TS at end of epoch
    '''

    def __init__(self, config_filepath):
        PrivCountClient.__init__(self, config_filepath)
        self.keystore = None

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
            self.keystore = state['keystore']

    def stopFactory(self):
        '''
        Called by twisted
        '''
        # TODO
        return
        if self.keystore is not None:
            # export everything that would be needed to survive an app restart
            state = {'keystore': self.keystore}
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

        logging.info("running share keeper using RSA public key id '{}'".format(self.config['name']))

        # connect to the tally server, register, and wait for commands
        self.do_checkin()
        reactor.run() # pylint: disable=E1101

    def get_status(self):
        '''
        Called by protocol
        Returns a dictionary containing status information
        '''
        return {'type':'ShareKeeper', 'name':self.config['name'],
        'state': 'active' if self.keystore is not None else 'idle', 'public_key':get_serialized_public_key(self.config['key'])}

    def do_checkin(self):
        '''
        Called by protocol
        Refresh the config, and try to connect to the server
        '''
        # TODO: Refactor common client code - issue #121
        self.refresh_config()
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

        if ('shares' not in config):
            logging.warning("start command from tally server cannot be completed due to missing shares")
            return None

        combined_counters = self.check_start_config(config)

        if combined_counters is None:
            return None
        else:
            config['counters'] = combined_counters

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
                logging.warning("failed to import blinding share {} config {}"
                                .format(share, config))
                # TODO: secure delete
                del private_key
                return None

        logging.info("successfully started and imported {} blinding shares for {} counters".format(len(share_list), len(config['counters'])))
        # TODO: secure delete
        del private_key
        return {}

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

        response_counts = None
        # send our counts
        if self.keystore is not None:
            response_counts = self.keystore.detach_counts()
        else:
            # let the TS decide what to do if the counts are missing
            logging.info("No keystore, counts never started")

        del self.keystore
        self.keystore = None

        return self.check_stop_config(config, response_counts)

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
            sk_conf = conf['share_keeper']

            # find the path for the secret handshake file
            sk_conf['secret_handshake'] = choose_secret_handshake_path(
                sk_conf, conf)

            # if key path is not specified, use default path
            if 'key' in sk_conf:
                sk_conf['key'] = normalise_path(sk_conf['key'])
            else:
                sk_conf['key'] = normalise_path('privcount.rsa_key.pem')
            # if the key does not exist, generate a new key
            if not os.path.exists(sk_conf['key']):
                generate_keypair(sk_conf['key'])

            sk_conf['name'] = get_public_digest(sk_conf['key'])

            # the state file
            sk_conf['state'] = normalise_path(sk_conf['state'])
            assert os.path.exists(os.path.dirname(sk_conf['state']))

            sk_conf['delay_period'] = self.get_valid_delay_period(sk_conf)

            sk_conf.setdefault('always_delay', False)
            assert isinstance(sk_conf['always_delay'], bool)

            sk_conf['sigma_decrease_tolerance'] = \
                self.get_valid_sigma_decrease_tolerance(sk_conf)

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
