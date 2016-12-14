#!/usr/bin/env python
'''
Created on Dec 12, 2015

@author: rob
'''
import sys
import argparse
import logging
from time import time

from twisted.internet import reactor
from twisted.internet.protocol import ServerFactory
from twisted.internet.error import ReactorNotRunning

from privcount.config import normalise_path
from privcount.connection import listen
from privcount.protocol import TorControlServerProtocol

DEFAULT_PRIVCOUNT_INJECT_SOCKET = '/tmp/privcount-inject'

listener = None

class PrivCountDataInjector(ServerFactory):

    def __init__(self, logpath, do_pause, prune_before, prune_after):
        self.logpath = logpath
        self.do_pause = do_pause
        self.prune_before = prune_before
        self.prune_after = prune_after
        self.protocol = None
        self.event_file = None
        self.last_time_end = 0.0

    def startFactory(self):
        # TODO
        return

    def stopFactory(self):
        # TODO
        return

    def buildProtocol(self, addr):
        # XXX multiple connections to our server will kill old connections
        self.protocol = TorControlServerProtocol(self)
        return self.protocol

    def start_injecting(self):
        if listener is not None:
            logging.info("Injector is no longer listening")
            listener.stopListening()
        if self.do_pause:
            logging.info("We will pause between the injection of each event to simulate actual event inter-arrival times, so this may take a while")

        if self.logpath == '-':
            self.event_file = sys.stdin
        else:
            self.event_file = open(normalise_path(self.logpath), 'r')
        self._inject_events()

    def stop_injecting(self):
        if self.event_file is not None:
            self.event_file.close()
            self.event_file = None

    def _get_line(self):
        if self.event_file == None:
            return None
        line = self.event_file.readline()
        if line == '':
            self.event_file.close()
            self.event_file = None
            return None
        else:
            return line

    def _flush_now(self, msg):
        if self.protocol is not None and self.protocol.transport is not None and self.protocol.transport.connected:
            # update event times so the data seems fresh to privcount
            this_time_start, this_time_end = self._get_event_times(msg)
            now = time()
            alive = this_time_end - this_time_start
            msg_adjusted_times = self._set_event_times(msg, now-alive, now)
            event = "650 {}".format(msg_adjusted_times)
            logging.info("sending event '{}'".format(event))
            self.protocol.sendLine(event)

    def _flush_later(self, msg):
        self._flush_now(msg)
        self._inject_events()

    def _inject_events(self):
        while True:
            line = self._get_line()
            if line is None:
                # close the connection from our server side
                if self.protocol is not None and self.protocol.transport is not None:
                    self.protocol.transport.loseConnection()
                try:
                    # we should no longer be listening, so the reactor should end here
                    reactor.stop()
                except ReactorNotRunning:
                    # it's ok if the reactor stopped before we told it to
                    pass
                return

            msg = line.strip()
            this_time_start, this_time_end = self._get_event_times(msg)

            # make sure this event is in our 'valid' event window
            if this_time_end < self.prune_before or this_time_end > self.prune_after:
                continue

            # if we need to simulate event inter-arrival times
            do_wait, wait_time = False, 0.0
            if self.do_pause:
                wait_time = this_time_end - self.last_time_end
                do_wait = True if self.last_time_end != 0.0 and wait_time > 0.0 else False
                self.last_time_end = this_time_end

            if do_wait:
                # we cant sleep or twisted wont work correctly, we must use callLater instead
                reactor.callLater(wait_time, self._flush_later, msg) # pylint: disable=E1101
                return
            else:
                self._flush_now(msg)

    def _get_event_times(self, msg):
        parts = msg.split()
        if parts[0] == 'PRIVCOUNT_CIRCUIT_ENDED' and len(parts) > 10:
            return float(parts[9]), float(parts[10])
        elif parts[0] == 'PRIVCOUNT_STREAM_ENDED' and len(parts) > 8:
            return float(parts[7]), float(parts[8])
        elif parts[0] == 'PRIVCOUNT_CONNECTION_ENDED' and len(parts) > 3:
            return float(parts[2]), float(parts[3])
        elif parts[0] == 'PRIVCOUNT_STREAM_BYTES_TRANSFERRED' and len(parts) > 6:
            return float(parts[6]), float(parts[6])
        return 0.0, 0.0

    def _set_event_times(self, msg, start_time, end_time):
        parts = msg.split()
        if parts[0] == 'PRIVCOUNT_CIRCUIT_ENDED' and len(parts) > 10:
            parts[9], parts[10] = start_time, end_time
        elif parts[0] == 'PRIVCOUNT_STREAM_ENDED' and len(parts) > 8:
            parts[7], parts[8] = start_time, end_time
        elif parts[0] == 'PRIVCOUNT_CONNECTION_ENDED' and len(parts) > 3:
            parts[2], parts[3] = start_time, end_time
        elif parts[0] == 'PRIVCOUNT_STREAM_BYTES_TRANSFERRED' and len(parts) > 6:
            parts[6] = end_time
        return ' '.join([str(p) for p in parts])

def main():
    ap = argparse.ArgumentParser(description="Injects Tor events into a PrivCount DC")
    add_inject_args(ap)
    args = ap.parse_args()
    run_inject(args)

def run_inject(args):
    '''
    start the injector, and start it listening
    '''
    # pylint: disable=E1101
    injector = PrivCountDataInjector(args.log, args.simulate, int(args.prune_before), int(args.prune_after))
    # The injector listens on all of IPv4, IPv6, and a control socket, and
    # injects events into the first client to connect
    # Since these are synthetic events, it is safe to use /tmp for the socket
    # path
    # XXX multiple connections to our server will kill old connections
    listener_config = {}
    if args.port is not None:
        listener_config['port'] = args.port
        if args.ip is not None:
            listener_config['ip'] = args.ip
    if args.unix is not None:
        listener_config['unix']= args.unix
    listeners = listen(injector, listener_config, ip_version_default = [4, 6])
    reactor.run()

def add_inject_args(parser):
    parser.add_argument('-p', '--port',
                        help="port on which to listen for PrivCount connections(default: no IP listener)",
                        required=False)
    parser.add_argument('-i', '--ip',
                        help="IPv4 or IPv6 address on which to listen for PrivCount connections (default: both 127.0.0.1 and ::1)",
                        required=False)
    parser.add_argument('-u', '--unix',
                        help="Unix socket on which to listen for PrivCount connections (default: '{}')"
                        .format(DEFAULT_PRIVCOUNT_INJECT_SOCKET),
                        required=False,
                        default=DEFAULT_PRIVCOUNT_INJECT_SOCKET)
    parser.add_argument('-l', '--log',
                        help="a file PATH to a PrivCount event log file, may be '-' for STDIN (default: STDIN)",
                        required=True,
                        default='-')
    parser.add_argument('-s', '--simulate',
                        action='store_true',
                        help="add pauses between each event injection to simulate the inter-arrival times from the source data",
                        default=True)
    parser.add_argument('--prune-before',
                        help="do not inject events that occurred before the given unix timestamp",
                        default=0)
    parser.add_argument('--prune-after',
                        help="do not inject events that occurred after the given unix timestamp",
                        default=2147483647)

if __name__ == "__main__":
    sys.exit(main())
