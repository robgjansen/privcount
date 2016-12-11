#!/usr/bin/env python
# Test that a local control port provides PrivCount events

import sys

from twisted.internet import reactor
from twisted.internet.protocol import ReconnectingClientFactory

from privcount.protocol import TorControlClientProtocol, get_valid_events

# Usage:
# cd privcount/tor
# ./configure && make check && make test-network-all
# src/or/tor PublishServerDescriptor 0 ControlPort 9051 ORPort 9001 DirPort 9030 ExitRelay 0 DataDirectory `mktemp -d`
# source venv/bin/activate
# python test/test_tor_ctl_event.py
# wait a few minutes for the first events to arrive

# The typical control port listed in the tor manual
TOR_CONTROL_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9051

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

    def buildProtocol(self, addr):
        '''
        Called by twisted
        '''
        client = TorControlClientProtocol(self)
        # ask for all the events
        client.startCollection(None, event_list=get_valid_events())
        return client

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
        event is a space-separated list of tokens from the event line
        '''
        if self.nickname is None:
            nickname = ""
        else:
            nickname = self.nickname + " "
        print "{}got PRIVCOUNT_* event: {}".format(nickname, " ".join(event))
        return True

reactor.connectTCP("127.0.0.1", TOR_CONTROL_PORT, TorCtlClient())
reactor.run()
