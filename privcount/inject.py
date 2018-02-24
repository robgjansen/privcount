#!/usr/bin/env python
'''
Created on Dec 12, 2015

@author: rob

See LICENSE for licensing information
'''
import sys
import argparse
import logging
from time import time

from twisted.internet import reactor, task
from twisted.internet.protocol import ServerFactory

from privcount.config import normalise_path
from privcount.connection import listen, stopListening
from privcount.data_collector import Aggregator
from privcount.log import errorCallback, stop_reactor, summarise_string
from privcount.protocol import TorControlServerProtocol
from privcount.tagged_event import parse_tagged_event, get_float_value

# We can't have the injector listen on a port by default, because it might
# conflict with a running tor instance
DEFAULT_PRIVCOUNT_INJECT_SOCKET = '/tmp/privcount-inject'

class PrivCountDataInjector(ServerFactory):

    def __init__(self, logpath, do_pause, prune_before, prune_after,
                 control_password = None, control_cookie_file = None):
        self.logpath = logpath
        self.do_pause = do_pause
        self.prune_before = prune_before
        self.prune_after = prune_after
        self.protocol = None
        self.event_file = None
        self.last_time_end = 0.0
        self.injecting = False
        self.pending_stop = False
        self.listeners = None
        self.control_password = control_password
        self.control_cookie_file = control_cookie_file
        self.input_line_count = 0
        self.output_line_count = 0
        self.output_event_count = 0

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

    def set_listeners(self, listener_list):
        '''
        Set the listeners for this factory to listener_list
        '''
        # Since the listeners hold a reference to the factory, this causes a
        # (temporary) reference loop. Perhaps there is a better way of
        # retrieving all the listeners for a factory?
        self.listeners = listener_list

    def get_control_password(self):
        '''
        Return the configured control password, or None if there is no control
        password.
        '''
        # Configuring multiple passwords is not supported
        return self.control_password

    def get_control_cookie_file(self):
        '''
        Return the configured control cookie file path, or None if there is no
        control cookie file.
        '''
        # Configuring multiple cookie files is not supported
        return self.control_cookie_file

    def start_injecting(self):
        self.injecting = True
        if self.listeners is not None:
            logging.info("Injector has connected: no longer listening for new connections")
            stopListening(self.listeners)
            # This breaks the reference loop
            self.listeners = None
        if self.do_pause:
            logging.info("We will pause between the injection of each event to simulate actual event inter-arrival times, so this may take a while")

        if self.logpath == '-':
            self.event_file = sys.stdin
        else:
            self.event_file = open(normalise_path(self.logpath), 'r')
        self._inject_events()

    def stop_injecting(self):
        '''
        Stop sending events to the data collector, close all connections.
        Depending on the protocol, closing the connection can erase all unread
        data. So we stop sending events, but delay the actual close for a
        short amount of time.
        This function sometimes reschedules itself using deferLater: any
        exceptions will be handled by errorCallback.
        '''
        if not self.injecting:
            logging.debug("Ignoring request to stop injecting when injection is not in progress")
            return
        self.injecting = False
        if not self.pending_stop:
            self.pending_stop = True
            logging.debug("Scheduling injector stop after a short delay")
            # twisted's loseConnection says:
            # "Close my connection, after writing all pending data"
            # But it often closes the connection before sending data.
            # So we allow the event loop to run before we stop injecting.
            # IP retains unread data in the network stack, but unix
            # sockets don't. So we need to delay closing unix sockets, for
            # the data collector to finish reading.
            # we don't care about cancelling here, we're stopping anyway
            stop_deferred = task.deferLater(reactor, 1.0,
                                            self.stop_injecting)
            stop_deferred.addErrback(errorCallback)
            return
        if self.listeners is not None:
            stopListening(self.listeners)
            self.listeners = None
        # Count lines and events
        event_info = ("Read {} lines, {} valid times, sent {} events"
                      .format(self.input_line_count, self.output_line_count,
                              self.output_event_count))
        # close the event log file
        if self.event_file is not None:
            self.event_file.close()
            self.event_file = None
            logging.warning("Connection closed before all events were sent. "
                            + event_info)
        else:
            logging.debug(event_info)
        # close the connection from our server side
        if self.protocol is not None and self.protocol.transport is not None:
            self.protocol.transport.loseConnection()
        # stop the reactor gracefully
        stop_reactor()

    def _get_line(self):
        if self.event_file == None:
            return None
        line = self.event_file.readline()
        if line == '':
            self.event_file.close()
            self.event_file = None
            return None
        else:
            self.input_line_count += 1
            return line

    def _flush_now(self, msg):
        if (self.protocol is not None and self.protocol.transport is not None
            and self.protocol.transport.connected and self.injecting):
            # update event times so the data seems fresh to privcount
            this_time_start, this_time_end = self._get_event_times(msg)
            now = time()
            alive = this_time_end - this_time_start
            msg_adjusted_times = self._set_event_times(msg, now-alive, now)
            event = "650 {}".format(msg_adjusted_times)
            self.output_event_count += 1
            logging.info("sending event {} '{}'"
                         .format(self.output_event_count,
                                 summarise_string(event)))
            logging.debug("sending event {} (full event) '{}'"
                          .format(self.output_event_count, event))
            self.protocol.sendLine(event)
        elif self.injecting:
            # No connection: stop sending
            self.stop_injecting()

    def _flush_later(self, msg):
        '''
        This function is called using deferLater, so any exceptions will be
        handled by errorCallback.
        '''
        self._flush_now(msg)
        self._inject_events()

    def _inject_events(self):
        while self.injecting:
            line = self._get_line()
            if line is None:
                # We're done
                self.stop_injecting()
                return

            msg = line.strip()
            this_time_start, this_time_end = self._get_event_times(msg)

            # make sure this event is in our 'valid' event window
            if this_time_end < self.prune_before or this_time_end > self.prune_after:
                logging.info("Event end time {} is outside {} to {}, skipping {}"
                             .format(this_time_end, self.prune_before,
                                     self.prune_after, msg))
                self.last_time_end = this_time_end
                continue

            self.output_line_count += 1

            # if we need to simulate event inter-arrival times
            wait_time = 0.0
            if self.do_pause:
                wait_time = this_time_end - self.last_time_end
                self.last_time_end = this_time_end

            # ensure wait_time is sensible
            if wait_time < 0.0:
                logging.warning("Out of sequence event times")
                wait_time = 0.0

            if wait_time < 2.0:
                logger = logging.debug
            elif wait_time < 60.0:
                logger = logging.info
            else:
                logger = logging.warning

            logger("Waiting {} seconds to send event {}".format(wait_time, msg))

            # we can't dump the entire file at once: it fills up the buffers
            # without giving the twisted event loop time to flush them
            # instead, use deferLater with a zero delay
            # we can't sleep or twisted won't work correctly
            inject_deferred = task.deferLater(reactor, wait_time,
                                              self._flush_later, msg)
            inject_deferred.addErrback(errorCallback)
            # _flush_later will inject the next event when called
            return

    START_TIMESTAMP_TAG = "CreatedTimestamp"
    END_TIMESTAMP_TAG = "EventTimestamp"

    def _get_event_times(self, msg):
        '''
        Return a tuple (start_time, end_time) for the event in msg.
        '''
        parts = msg.split()
        if len(parts) > 0:
            event_desc = "in _get_event_times for {} event".format(parts[0])
        else:
            event_desc = "in _get_event_times"
        # Positional Events: these events have hard-coded field orders
        if parts[0] == 'PRIVCOUNT_STREAM_BYTES_TRANSFERRED' and len(parts) == Aggregator.STREAM_BYTES_ITEMS + 1:
            return float(parts[6]), float(parts[6])
        elif parts[0] == 'PRIVCOUNT_STREAM_ENDED' and len(parts) == Aggregator.STREAM_ENDED_ITEMS + 1:
            return float(parts[7]), float(parts[8])
        elif parts[0] == 'PRIVCOUNT_CIRCUIT_ENDED' and len(parts) == Aggregator.CIRCUIT_ENDED_ITEMS + 1:
            return float(parts[7]), float(parts[8])
        elif parts[0] == 'PRIVCOUNT_CONNECTION_ENDED' and len(parts) == Aggregator.CONNECTION_ENDED_ITEMS + 1:
            return float(parts[2]), float(parts[3])
        # Tagged Events: these events have fields with Key=Value pairs
        # All current tagged events have at least one field
        # Handles:
        #  - 'PRIVCOUNT_CIRCUIT_CELL'
        #  - 'PRIVCOUNT_CIRCUIT_CLOSE'
        #  - 'PRIVCOUNT_CONNECTION_CLOSE'
        #  - 'PRIVCOUNT_HSDIR_CACHE_STORE'
        #  - 'PRIVCOUNT_HSDIR_CACHE_FETCH'
        #  - 'PRIVCOUNT_VITERBI'
        elif '=' in msg and len(parts) >= 2:
            items = parts[1:]
            fields = parse_tagged_event(items)
            # malformed
            if len(items) > 0 and len(fields) == 0:
                logging.warning("Malformed tagged event in: '{}' {}"
                                .format(msg, event_desc))
            else:
                # PRIVCOUNT_CIRCUIT_CLOSE and PRIVCOUNT_CONNECTION_CLOSE have
                # start and end times
                start_time = get_float_value(
                                    PrivCountDataInjector.START_TIMESTAMP_TAG,
                                    fields, event_desc,
                                    is_mandatory=False)

                # PRIVCOUNT_CIRCUIT_CELL, PRIVCOUNT_HSDIR_CACHE_STORE,
                # PRIVCOUNT_HSDIR_CACHE_FETCH, and PRIVCOUNT_VITERBI
                #  have a single event time: the "end" time
                # (CACHE_FETCH has CacheCreatedTime, which we don't update:
                # it's not the start time, and it's not at the same resolution)
                end_time = get_float_value(
                                    PrivCountDataInjector.END_TIMESTAMP_TAG,
                                    fields, event_desc,
                                    is_mandatory=True)

                # use the end time as a dummy start time for single-time
                # events
                if start_time is None:
                    start_time = end_time

                return start_time, end_time
        else:
            logging.warning("Wrong event field count or unknown or empty event in: '{}' {}"
                            .format(msg, event_desc))
        return 0.0, 0.0

    def _set_event_times(self, msg, start_time, end_time):
        '''
        Set the event times in msg to start_time and end_time, and return
        the modified event. May change the field order of tagged events.
        '''
        parts = msg.split()
        if len(parts) > 0:
            event_desc = "in _set_event_times for {} event".format(parts[0])
        else:
            event_desc = "in _set_event_times"
        # Positional Events: these events have hard-coded field orders
        if parts[0] == 'PRIVCOUNT_STREAM_BYTES_TRANSFERRED' and len(parts) == Aggregator.STREAM_BYTES_ITEMS + 1:
            parts[6] = end_time
        elif parts[0] == 'PRIVCOUNT_STREAM_ENDED' and len(parts) == Aggregator.STREAM_ENDED_ITEMS + 1:
            parts[7], parts[8] = start_time, end_time
        elif parts[0] == 'PRIVCOUNT_CIRCUIT_ENDED' and len(parts) == Aggregator.CIRCUIT_ENDED_ITEMS + 1:
            parts[7], parts[8] = start_time, end_time
        elif parts[0] == 'PRIVCOUNT_CONNECTION_ENDED' and len(parts) == Aggregator.CONNECTION_ENDED_ITEMS + 1:
            parts[2], parts[3] = start_time, end_time
        # Tagged Events: these events have fields with tags, order is irrelevant
        # All current tagged events have at least one field
        # Handles:
        #  - 'PRIVCOUNT_CIRCUIT_CELL'
        #  - 'PRIVCOUNT_CIRCUIT_CLOSE'
        #  - 'PRIVCOUNT_HSDIR_CACHE_STORE'
        #  - 'PRIVCOUNT_VITERBI'
        elif '=' in msg and len(parts) >= 2:
            items = parts[1:]
            fields = parse_tagged_event(items)
            # malformed
            if len(items) > 0 and len(fields) == 0:
                logging.warning("Malformed tagged event in: '{}' {}"
                                .format(msg, event_desc))
            else:
                # PRIVCOUNT_CIRCUIT_CLOSE has start and end times
                if PrivCountDataInjector.START_TIMESTAMP_TAG in fields:
                    fields[PrivCountDataInjector.START_TIMESTAMP_TAG] = str(start_time)

                # PRIVCOUNT_CIRCUIT_CELL and PRIVCOUNT_HSDIR_CACHE_STORE have
                # a single event time: the "end" time
                fields[PrivCountDataInjector.END_TIMESTAMP_TAG] = str(end_time)

                # This reorders the fields in key hash order
                # This is not a stable order: it may change between python
                # versions. And clients should not depend on the exact order
                # anyway.
                fields_list = ["{}={}".format(k, fields[k]) for k in fields]
                parts_list = [parts[0]] + fields_list

                return ' '.join(parts_list)
        else:
            logging.warning("Wrong event field count or unknown or empty event in: '{}' {}"
                            .format(msg, event_desc))
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
    injector = PrivCountDataInjector(args.log, args.simulate, float(args.prune_before), float(args.prune_after), args.control_password, args.control_cookie_file)
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
    injector.set_listeners(listeners)
    reactor.run()

def add_inject_args(parser):
    parser.add_argument('-p', '--port',
                        help="port on which to listen for PrivCount connections(default: no IP listener)",
                        required=False)
    parser.add_argument('-i', '--ip',
                        help="IPv4 or IPv6 address on which to listen for PrivCount connections (default: both 127.0.0.1 and ::1, if a port is specified)",
                        required=False)
    parser.add_argument('-u', '--unix',
                        help="Unix socket on which to listen for PrivCount connections (default: no unix listener)",
                        required=False)
    parser.add_argument('-l', '--log',
                        help="a file PATH to a PrivCount event log file, may be '-' for STDIN (default: STDIN)",
                        required=True,
                        default='-')
    parser.add_argument('-s', '--simulate',
                        action='store_true',
                        help="add pauses between each event injection to simulate the inter-arrival times from the source data")
    parser.add_argument('--prune-before',
                        help="do not inject events that occurred before the given unix timestamp",
                        default=float(0))
    parser.add_argument('--prune-after',
                        help="do not inject events that occurred after the given unix timestamp",
                        default=float(sys.maxint))
    parser.add_argument('--control-password',
                        help="A file containing the tor control password. Set this in tor using tor --hash-password and HashedControlPassword")
    parser.add_argument('--control-cookie-file',
                        help="The tor control cookie file. Set this in tor using CookieAuthentication and CookieAuthFile")

if __name__ == "__main__":
    sys.exit(main())
