#!/usr/bin/env python
# Test that a local control port (IP or unix socket) provides PrivCount events

import logging
import sys

from collections import Sequence
from os import path

from twisted.internet import reactor
from twisted.internet.protocol import ReconnectingClientFactory

from privcount.connection import connect, transport_info, get_a_control_password
from privcount.protocol import TorControlClientProtocol, get_valid_events

# set the log level
#logging.basicConfig(level=logging.DEBUG)

## Usage:
#
## Setup
# cd privcount/tor
# ./configure && make check && make test-network-all
#
## Control Socket (relies on filesystem security, more secure)
# mkdir -p /var/run/tor
# chmod go-rwx /var/run/tor
# src/or/tor PublishServerDescriptor 0 ORPort 9001 DirPort 9030 ExitRelay 0 DataDirectory `mktemp -d` ControlSocket /var/run/tor/control
#
## OR
#
## Control Port (any local process can access the control port, less secure)
# src/or/tor PublishServerDescriptor 0 ORPort 9001 DirPort 9030 ExitRelay 0 DataDirectory `mktemp -d` ControlPort 9051
#
## Testing
# source venv/bin/activate
# python test/test_tor_ctl_event.py [ unix_socket_path | control_port |
#                                     control_password ]
## wait a few minutes for the first events to arrive, or just terminate tor
## using a SIGINT after it has made some connections

# The default control socket path used by Debian
TOR_CONTROL_PATH = '/var/run/tor/control'

# The typical control port listed in the tor manual
TOR_CONTROL_PORT = 9051
# Try localhost on this IP version
TOR_CONTROL_IP_VERSION = 4

config_list = []

class TorCtlClient(ReconnectingClientFactory):
    '''
    Connects to a local Tor Control port to test if events of type
    'PRIVCOUNT_*' are properly exported by Tor.
    '''

    def __init__(self):
        '''
        Make sure the stored nickname is initialised
        '''
        self.nickname = None
        self.client = None

    def buildProtocol(self, addr):
        '''
        Called by twisted
        '''
        self.client = TorControlClientProtocol(self)
        # ask for all the events
        self.client.startCollection(None, event_list=get_valid_events())
        return self.client

    def get_control_password(self):
        '''
        Return the configured control password, or None if no connections have
        a control password.
        '''
        # Configuring multiple passwords is not supported
        return get_a_control_password(config_list)

    def set_nickname(self, nickname):
        '''
        Called when the relay provides its nickname
        '''
        self.nickname = nickname
        return True

    def set_fingerprint(self, fingerprint):
        '''
        Called when the relay provides its fingerprint
        '''
        print "Relay fingerprint: {}".format(fingerprint)
        if self.nickname is not None:
            print "Relay nickname: {}".format(self.nickname)
        return True

    def handle_event(self, event):
        '''
        Called when an event occurs.
        event is a list of tokens from the event line, split on spaces
        '''
        if self.nickname is None:
            nickname = ""
        else:
            nickname = self.nickname + " "
        address = transport_info(self.client.transport)
        print "{}{} got PRIVCOUNT_* event: {}".format(nickname,
                                                      address,
                                                      " ".join(event))
        return True

# Process command-line arguments
if len(sys.argv) > 1:
    # Just try every possible method, and let the ones that don't work fail
    # (if they all succeed, it will be terribly confusing)
    # Don't use multiple configs in production, it's only useful for testing
    for arg in sys.argv[1:]:
        # UNIX socket path
        if path.exists(arg):
            logging.warning("Trying unix socket: {}".format(arg))
            path_config = { 'unix' : arg }
            config_list.append(path_config)
        # IP port
        try:
            control_port = int(arg)
            # We could use TOR_CONTROL_IP_VERSION to be more specific about
            # exactly which localhost we're trying
            logging.warning("Trying localhost port: {}".format(control_port))
            port_config = { 'port' : control_port }
            config_list.append(port_config)
        except ValueError:
            # Ok, so maybe it's a password?
            # TODO: use actual --argument-name specifiers
            control_password = arg
            logging.warning("Trying control password: {}"
                            .format(control_password))
            password_config = { 'control_password' : control_password }
            # the controller picks an arbitrary password, so don't supply more
            # than one
            config_list.append(password_config)

# Set defaults if there is nothing in the config list
if len(config_list) == 0:
    path_config = { 'unix' : TOR_CONTROL_PATH }
    config_list.append(path_config)
    port_config = { 'port' : TOR_CONTROL_PORT }
    config_list.append(port_config)

# Trying localhost is sufficient: a service bound to all ports will answer
# on localhost (and expose control of your tor to the entire world).
connector = connect(TorCtlClient(),
                    config_list,
                    ip_version_default = TOR_CONTROL_IP_VERSION)

if isinstance(connector, (Sequence)):
    logging.warning("If you have ControlPorts listening on multiple IP ports or unix paths, some events may be attributed to the wrong address, or duplicated.")
reactor.run()
