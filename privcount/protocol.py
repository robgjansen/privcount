import random, logging, json

from time import time
from os import _exit

from twisted.internet import reactor
from twisted.protocols.basic import LineOnlyReceiver

PRIVCOUNT_HANDSHAKE_MAGIC = 759.623

class PrivCountProtocol(LineOnlyReceiver):
    '''
    The base protocol class for PrivCount. This class logs basic connection
    information when connections are made and lost, and tracks the validity of
    connections during the handshake process and execution of the protocol.
    '''

    def __init__(self, factory):
        self.factory = factory
        self.is_valid_connection = False
        self.client_cookie = None
        self.server_cookie = None

        '''here we use the LineOnlyReceiver's MAX_LENGTH to drop the connection
        if we receive too much data before the handshake validates it
        the handshake process itself transfers very little, so we can get
        away with a small buffer - after the handshake suceeds, we allow lines
        of longer length'''
        self.MAX_LENGTH = 256

    def connectionMade(self): # overrides twisted function
        peer = self.transport.getPeer()
        logging.debug("Connection with {}:{}:{} was made".format(peer.type, peer.host, peer.port))

    def lineReceived(self, line): # overrides twisted function
        peer = self.transport.getPeer()
        logging.debug("Received line '{}' from {}:{}:{}".format(line, peer.type, peer.host, peer.port))
        parts = [part.strip() for part in line.split(' ', 1)]
        if len(parts) > 0:
            event_type = parts[0]
            event_payload = parts[1] if len(parts) > 1 else ''
            self.process_event(event_type, event_payload)

    def sendLine(self, line): # overrides twisted function
        peer = self.transport.getPeer()
        logging.debug("Sending line '{}' to {}:{}:{}".format(line, peer.type, peer.host, peer.port))
        return LineOnlyReceiver.sendLine(self, line)

    def lineLengthExceeded(self, line): # overrides twisted function
        peer = self.transport.getPeer()
        logging.warning("Incomming line of length {} exceeded MAX_LENGTH of {}, dropping unvalidated connection to {}:{}:{}".format(len(line), self.MAX_LENGTH, peer.type, peer.host, peer.port))
        return LineOnlyReceiver.lineLengthExceeded(self, line)

    def process_event(self, event_type, event_payload):
        try:
            if event_type.startswith('HANDSHAKE'):
                self.handle_handshake_event(event_type, event_payload)
                return

            if self.is_valid_connection:
                is_valid = False

                if event_type.startswith('STATUS'):
                    is_valid = self.handle_status_event(event_type,
                                                        event_payload)
                elif event_type.startswith('START'):
                    is_valid = self.handle_start_event(event_type,
                                                       event_payload)
                elif event_type.startswith('STOP'):
                    is_valid = self.handle_stop_event(event_type,
                                                      event_payload)
                elif event_type.startswith('CHECKIN'):
                    is_valid = self.handle_checkin_event(event_type,
                                                         event_payload)
                self.is_valid_connection = is_valid

        except BaseException as e:
            # Any unhandled exception in client/server code is an error
            # that should terminate the connection. Otherwise, a client
            # failing due to an exception can cause an infinite retry loop.
            logging.error(
                "Exception {} while processing event type: {} payload: {}"
                .format(e, event_type, event_payload))

            # That said, terminating on exception is a denial of service
            # risk: if an untrusted party can cause an exception, they can
            # bring down the privcount network.
            # Since we ignore unrecognised events, the client would have to be
            # maliciously crafted, not just a port scanner.

            # die immediately using os._exit()
            # we can't use sys.exit() here, because twisted catches and logs it
            _exit(1)

        if not self.is_valid_connection:
            self.protocol_failed()

    def handshake_succeeded(self):
        peer = self.transport.getPeer()
        logging.debug("Handshake with {}:{}:{} was successful".format(peer.type, peer.host, peer.port))
        self.is_valid_connection = True
        self.MAX_LENGTH = 512*1024 # now allow longer lines

    def handshake_failed(self):
        peer = self.transport.getPeer()
        logging.warning("Handshake with {}:{}:{} failed".format(peer.type, peer.host, peer.port))
        self.is_valid_connection = False
        self.transport.loseConnection()

    def protocol_succeeded(self):
        peer = self.transport.getPeer()
        logging.debug("Protocol with {}:{}:{} was successful".format(peer.type, peer.host, peer.port))
        self.transport.loseConnection()

    def protocol_failed(self):
        peer = self.transport.getPeer()
        logging.warning("Protocol with {}:{}:{} failed".format(peer.type, peer.host, peer.port))
        self.is_valid_connection = False
        self.transport.loseConnection()

    def connectionLost(self, reason): # overrides twisted function
        peer = self.transport.getPeer()
        logging.debug("Connection with {}:{}:{} was lost: {}".format(peer.type, peer.host, peer.port, reason.getErrorMessage()))

    def handle_handshake_event(self, event_type, event_payload):
        '''
        override this function in a subclass to handle the event type
        '''
        pass

    def handle_status_event(self, event_type, event_payload):
        '''
        override this function in a subclass to handle the event type
        '''
        pass

    def handle_start_event(self, event_type, event_payload):
        '''
        override this function in a subclass to handle the event type
        '''
        pass

    def handle_stop_event(self, event_type, event_payload):
        '''
        override this function in a subclass to handle the event type
        '''
        pass

    def handle_checkin_event(self, event_type, event_payload):
        '''
        override this function in a subclass to handle the event type
        '''
        pass

class PrivCountServerProtocol(PrivCountProtocol):

    def __init__(self, factory):
        PrivCountProtocol.__init__(self, factory)
        self.last_sent_time = 0.0
        self.client_uid = None

    def connectionMade(self): # overrides twisted function
        PrivCountProtocol.connectionMade(self)
        self.send_handshake_event()

    def send_handshake_event(self):
        '''
        initiate the handshake with the client
        '''
        self.server_cookie = round(random.random(), 6)
        self.sendLine("HANDSHAKE1 {}".format(self.server_cookie))

    def handle_handshake_event(self, event_type, event_payload):
        is_valid = False
        parts = event_payload.strip().split()

        if event_type == "HANDSHAKE2" and len(parts) == 2:
            is_valid = True
            self.client_cookie = float(parts[0])
            client_password = float(parts[1])
            password = round(self.client_cookie * self.server_cookie * PRIVCOUNT_HANDSHAKE_MAGIC, 6)
            if client_password == float(str(password)):
                self.sendLine("HANDSHAKE3 SUCCESS")
                self.handshake_succeeded()
            else:
                self.sendLine("HANDSHAKE3 FAIL")
                self.handshake_failed()

        if not is_valid:
            self.handshake_failed()

    def handshake_succeeded(self):
        PrivCountProtocol.handshake_succeeded(self)
        self.send_status_event()

    def send_status_event(self):
        status = self.factory.get_status()
        self.sendLine("STATUS {} {}".format(time(), json.dumps(status)))
        self.last_sent_time = time()

    def handle_status_event(self, event_type, event_payload):
        parts = event_payload.split(' ', 1)

        if event_type == "STATUS" and len(parts) == 2:
            client_status = json.loads(parts[1])

            client_status['alive'] = time()
            client_status['host'] = self.transport.getPeer().host
            client_status['clock_skew'] = 0.0
            client_status['rtt'] = 0.0

            if self.last_sent_time > 0:
                client_status['rtt'] = client_status['alive'] - self.last_sent_time
                latency = client_status['rtt'] / 2.0
                client_time = float(parts[0])
                client_status['clock_skew'] = abs(time() - latency - client_time)
                self.last_sent_time = 0

            self.client_uid = "{}~{}".format(client_status['host'], client_status['name'])
            self.factory.set_client_status(self.client_uid, client_status)

            config = self.factory.get_stop_config(self.client_uid)
            if config is not None:
                self.send_stop_event(config)
            else:
                config = self.factory.get_start_config(self.client_uid)
                if config is not None:
                    self.send_start_event(config)
                else:
                    self.send_checkin_event()
            return True

        return False

    def send_start_event(self, config):
        assert config is not None
        self.sendLine("START {}".format(json.dumps(config)))

    def handle_start_event(self, event_type, event_payload):
        parts = event_payload.split(' ', 1)
        if event_type == "START" and len(parts) > 0:
            result_data = None
            if parts[0] == "SUCCESS" and len(parts) == 2:
                result_data = json.loads(parts[1])
            self.factory.set_start_result(self.client_uid, result_data)
            self.send_status_event()
            return True
        else:
            return False

    def send_stop_event(self, config):
        assert config is not None
        self.sendLine("STOP {}".format(json.dumps(config)))

    def handle_stop_event(self, event_type, event_payload):
        parts = event_payload.split(' ', 1)
        if event_type == "STOP" and len(parts) > 0:
            result_data = None
            if parts[0] == "SUCCESS" and len(parts) == 2:
                result_data = json.loads(parts[1])
            self.factory.set_stop_result(self.client_uid, result_data)
            self.send_status_event()
            return True
        else:
            return False

    def send_checkin_event(self):
        period = int(self.factory.get_checkin_period())
        self.sendLine("CHECKIN {}".format(period))

    def handle_checkin_event(self, event_type, event_payload):
        self.protocol_succeeded()
        return True

class PrivCountClientProtocol(PrivCountProtocol):

    def handshake_succeeded(self):
        PrivCountProtocol.handshake_succeeded(self)
        # for a reconnecting client, reset the exp backoff delay
        self.factory.resetDelay()

    def protocol_succeeded(self):
        PrivCountProtocol.protocol_succeeded(self)
        # for a reconnecting client, don't reconnect after this disconnection
        self.factory.stopTrying()

    def handle_handshake_event(self, event_type, event_payload):
        is_valid = False
        parts = event_payload.split()

        if event_type == "HANDSHAKE1" and len(parts) == 1:
            is_valid = True
            self.server_cookie = float(parts[0])
            self.client_cookie = round(random.random(), 6)
            password = round(self.client_cookie * self.server_cookie * PRIVCOUNT_HANDSHAKE_MAGIC, 6)
            self.sendLine("HANDSHAKE2 {} {}".format(self.client_cookie, password))
        elif event_type == "HANDSHAKE3" and len(parts) == 1:
            is_valid = True
            if parts[0] == "SUCCESS":
                self.handshake_succeeded()
            else:
                self.handshake_failed()

        if not is_valid:
            self.handshake_failed()

    def handle_status_event(self, event_type, event_payload):
        parts = event_payload.split(' ', 1)

        if event_type == "STATUS" and len(parts) == 2:
            server_time = float(parts[0])
            server_status = json.loads(parts[1])
            self.factory.set_server_status(server_status)

            status = self.factory.get_status()
            self.sendLine("STATUS {} {}".format(time(), json.dumps(status)))
            return True
        return False

    def handle_start_event(self, event_type, event_payload):
        start_config = json.loads(event_payload)
        result_data = self.factory.do_start(start_config)
        if result_data is not None:
            self.sendLine("START SUCCESS {}".format(json.dumps(result_data)))
        else:
            self.sendLine("START FAIL")
        return True

    def handle_stop_event(self, event_type, event_payload):
        stop_config = json.loads(event_payload)
        result_data = self.factory.do_stop(stop_config)
        if result_data is not None:
            self.sendLine("STOP SUCCESS {}".format(json.dumps(result_data)))
        else:
            self.sendLine("STOP FAIL")
        return True

    def handle_checkin_event(self, event_type, event_payload):
        if event_type == "CHECKIN":
            parts = event_payload.split()
            if len(parts) == 1:
                period = int(parts[0])
                reactor.callLater(period, self.factory.do_checkin) # pylint: disable=E1101
                self.sendLine("CHECKIN SUCCESS")
                self.protocol_succeeded()
                return True
        return False

class TorControlClientProtocol(LineOnlyReceiver):

    def __init__(self, factory):
        self.factory = factory
        self.state = None

    def connectionMade(self): # overrides twisted function
        peer = self.transport.getPeer()
        logging.debug("Connection with {}:{}:{} was made".format(peer.type, peer.host, peer.port))

        self.sendLine("AUTHENTICATE")
        self.state = 'authenticating'

    def lineReceived(self, line): # overrides twisted function
        peer = self.transport.getPeer()
        line = line.strip()
        logging.debug("Received line '{}' from {}:{}:{}".format(line, peer.type, peer.host, peer.port))

        if self.state == 'authenticating' and line == "250 OK":
            # Just ask for all the info all at once
            self.sendLine("GETCONF Nickname")
            self.sendLine("GETCONF ORPort")
            self.sendLine("GETCONF DirPort")
            self.sendLine("GETINFO version")
            self.sendLine("GETINFO address")
            self.sendLine("GETINFO fingerprint")
            self.state = 'discovering'
        elif self.state == 'discovering':
            # -- These are the continuing cases, they can continue discovering, skip_ok, or quit() --
            # It's a relay, and it's just told us its Nickname
            if line.startswith("250 Nickname="):
                _, _, nickname = line.partition("Nickname=") # returns: part1, separator, part2
                if not self.factory.set_nickname(nickname):
                    logging.warning("Connection with {}:{}:{}: bad nickname {}".format(peer.type, peer.host, peer.port, nickname))
            # It doesn't have a Nickname, maybe it's a client?
            # But we'll catch that when we check the fingerprint, so just ignore this response
            elif line == "250 Nickname":
                logging.info("Connection with {}:{}:{}: no Nickname".format(peer.type, peer.host, peer.port))
            # It's a relay, and it's just told us its ORPort
            elif line.startswith("250 ORPort="):
                _, _, orport = line.partition("ORPort=") # returns: part1, separator, part2
                if not self.factory.set_orport(orport):
                    logging.warning("Connection with {}:{}:{}: bad ORPort {}".format(peer.type, peer.host, peer.port, orport))
            # It doesn't have an ORPort, maybe it's a client?
            # But we'll catch that when we check the fingerprint, so just ignore this response
            elif line == "250 ORPort":
                logging.warning("Connection with {}:{}:{}: no ORPort".format(peer.type, peer.host, peer.port))
            # It's a relay, and it's just told us its DirPort
            elif line.startswith("250 DirPort="):
                _, _, dirport = line.partition("DirPort=") # returns: part1, separator, part2
                if not self.factory.set_dirport(dirport):
                    logging.warning("Connection with {}:{}:{}: bad DirPort {}".format(peer.type, peer.host, peer.port, dirport))
            # It doesn't have an DirPort, just ignore the response
            elif line == "250 DirPort":
                logging.info("Connection with {}:{}:{}: no DirPort".format(peer.type, peer.host, peer.port))
            # It's just told us its version
            # The control spec assumes that Tor always has a version, so there's no error case
            elif line.startswith("250-version="):
                _, _, version = line.partition("version=") # returns: part1, separator, part2
                if not self.factory.set_version(version):
                    logging.warning("Connection with {}:{}:{}: bad version {}".format(peer.type, peer.host, peer.port, version))
                self.state = 'skip_ok'
            # It's just told us its address
            elif line.startswith("250-address="):
                _, _, address = line.partition("address=") # returns: part1, separator, part2
                if not self.factory.set_address(address):
                    logging.warning("Connection with {}:{}:{}: bad address {}".format(peer.type, peer.host, peer.port, address))
                self.state = 'skip_ok'
            # We asked for its address, and it couldn't find it. That's weird.
            elif line == "551 Address unknown":
                logging.info("Connection with {}:{}:{}: does not know its own address".format(peer.type, peer.host, peer.port))
            # -- These are the terminating cases, they must end in processing or quit() --
            # It's a relay, and it's just told us its fingerprint
            elif line.startswith("250-fingerprint="):
                _, _, fingerprint = line.partition("fingerprint=") # returns: part1, separator, part2
                if not self.factory.set_fingerprint(fingerprint):
                    logging.warning("Connection with {}:{}:{}: bad fingerprint {}".format(peer.type, peer.host, peer.port, fingerprint))
                # processing mode will skip any unrecognised lines, such as "250 OK"
                self.state = 'processing'
            # We asked for its fingerprint, and it said it's a client
            elif line == "551 Not running in server mode":
                logging.warning("Connection with {}:{}:{} failed: not a relay".format(peer.type, peer.host, peer.port))
                self.quit()
            # something unexpectedly bad happened
            elif line.startswith("5"):
                logging.warning("Connection with {}:{}:{} failed: unexpected error response: '{}'".format(peer.type, peer.host, peer.port, line))
                self.quit()
            # something unexpected, but ok happened
            elif line.startswith("2"):
                logging.info("Connection with {}:{}:{}: unexpected OK response: '{}'".format(peer.type, peer.host, peer.port, line))
                self.state = 'processing'
            # something unexpected happened, let's assume it's ok
            else:
                logging.warning("Connection with {}:{}:{} failed: unexpected response: '{}'".format(peer.type, peer.host, peer.port, line))
                self.state = 'processing'

            # we're done with discovering context, let's start processing events
            if self.state == 'processing':
                self.sendLine("SETEVENTS PRIVCOUNT")
        elif self.state == 'skip_ok':
            # just skip one OK line
            if line == "250 OK":
                self.state = 'discovering'
            else:
                logging.warning("Connection with {}:{}:{} failed: unexpected response: '{}'".format(peer.type, peer.host, peer.port, line))
                self.quit()
        elif self.state == 'processing' and line.startswith("650 PRIVCOUNT "):
            _, _, event = line.partition(" PRIVCOUNT ") # returns: part1, separator, part2
            if event != '':
                if not self.factory.handle_event(event):
                    self.quit()
        # log any non-privcount responses at an appropriate level
        elif line == "250 OK":
            logging.debug("Connection with {}:{}:{}: ok response: '{}'".format(peer.type, peer.host, peer.port, line))
        elif self.state == 'processing' and line.startswith("5"):
            logging.warning("Connection with {}:{}:{}: unexpected response: '{}'".format(peer.type, peer.host, peer.port, line))
        elif self.state == 'processing' and line.startswith("2"):
            logging.info("Connection with {}:{}:{}: ok response: '{}'".format(peer.type, peer.host, peer.port, line))

    def quit(self):
        self.sendLine("QUIT")

    def connectionLost(self, reason): # overrides twisted function
        peer = self.transport.getPeer()
        logging.debug("Connection with {}:{}:{} was lost: {}".format(peer.type, peer.host, peer.port, reason.getErrorMessage()))

class TorControlServerProtocol(LineOnlyReceiver):
    '''
    The server side of the Tor control protocol as exercised by PrivCount.

    This is useful for emulating a Tor control server for testing purposes.

    Alternately, run a temporary, non-publishing relay with:
    tor DataDirectory /tmp/tor.$$ ORPort 12345 PublishServerDescriptor 0 ControlSocket /tmp/tor.$$/control_socket
    And use stem's tor-promt to query it:
    ./tor-prompt -s /tmp/tor.$$/control_socket
    (where $$ is the PID of the shell tor is running in.)

    Example protocol run:
    telnet localhost 9051
    Connected to localhost.
    Escape character is '^]'.
    AUTHENTICATE
    250 OK
    GETINFO
    250 OK
    GETINFO fingerprint
    250-fingerprint=5E54527B88A544E1A6CBF412A1DE2B208E7BF9A2
    250 OK
    GETINFO blah
    552 Unrecognized key "blah"
    GETCONF Nickname
    250 Nickname=Unnamed
    GETCONF foo
    552 Unrecognized configuration key "foo"
    SETEVENTS PRIVCOUNT
    552 Unrecognized event "PRIVCOUNT"
    SETEVENTS BW
    250 OK
    650 BW 422670 576069
    QUIT
    250 closing connection
    Connection closed by foreign host.
    '''

    def __init__(self, factory):
        self.factory = factory
        self.authenticated = False

    def connectionMade(self):
        '''
        overrides twisted function
        '''
        peer = self.transport.getPeer()
        logging.debug("Connection with {}:{}:{} was made".format(peer.type, peer.host, peer.port))

    def lineReceived(self, line):
        '''
        overrides twisted function
        '''
        peer = self.transport.getPeer()
        line = line.strip()
        parts = line.split(' ')

        logging.debug("Received line '{}' from {}:{}:{}".format(line, peer.type, peer.host, peer.port))

        if not self.authenticated:
            if line == "AUTHENTICATE":
                self.sendLine("250 OK")
                self.authenticated = True
            else:
                self.sendLine("514 Authentication required.")
                self.transport.loseConnection()
        elif len(parts) > 0:
            if parts[0] == "SETEVENTS":
                if len(parts) == 2:
                    if parts[1] == "PRIVCOUNT":
                        self.sendLine("250 OK")
                        self.factory.start_injecting()
                    else:
                        self.sendLine('552 Unrecognized event "{}"'.format(parts[1]))
                        self.factory.stop_injecting()
                else:
                    self.sendLine('552 Unrecognized event ""')
            elif parts[0] == "GETINFO":
                # the correct response to an empty GETINFO is an empty OK
                if len(parts) == 1:
                    self.sendLine("250 OK")
                # GETINFO is case-sensitive, and returns multiple lines
                # The control spec says we should say "Tor ", but tor doesn't
                elif len(parts) == 2 and parts[1] == "version":
                    self.sendLine("250-version=0.2.8.6 (git-4d217548e3f05569)")
                    self.sendLine("250 OK")
                elif len(parts) == 2 and parts[1] == "address":
                    # TEST-NET from https://tools.ietf.org/html/rfc5737
                    self.sendLine("250-address=192.0.2.91")
                    self.sendLine("250 OK")
                elif len(parts) == 2 and parts[1] == "fingerprint":
                    self.sendLine("250-fingerprint=FACADE0000000000000000000123456789ABCDEF")
                    self.sendLine("250 OK")
                else:
                    # strictly, GETINFO should process each word on the line
                    # as a separate request for information.
                    # But this is sufficient for our purposes, and is still
                    # more-or-less conformant to the control spec.
                    self.sendLine('552 Unrecognized key "{}"'.format(parts[1]))
            elif parts[0] == "GETCONF":
                # the correct response to an empty GETCONF is an empty OK
                if len(parts) == 1:
                    self.sendLine("250 OK")
                # Unlike GETINFO, GETCONF is case-insensitive, and returns one line
                # It also uses the canonical case of the option in its response
                elif len(parts) == 2 and parts[1].lower() == "nickname":
                    self.sendLine("250 Nickname=PrivCountTorRelay99")
                elif len(parts) == 2 and parts[1].lower() == "orport":
                    self.sendLine("250 ORPort=9001")
                elif len(parts) == 2 and parts[1].lower() == "dirport":
                    self.sendLine("250 DirPort=9030")
                else:
                    # Like GETINFO, our GETCONF does not accept multiple words
                    self.sendLine('552 Unrecognized configuration key "{}"'.format(parts[1]))
            elif parts[0] == "QUIT":
                self.factory.stop_injecting()
                self.sendLine("250 closing connection")
                self.transport.loseConnection()
            else:
                self.sendLine('510 Unrecognized command "{}"'.format(parts[0]))
        else:
            self.sendLine('510 Unrecognized command ""')

    def connectionLost(self, reason):
        '''
        overrides twisted function
        '''
        peer = self.transport.getPeer()
        logging.debug("Connection with {}:{}:{} was lost: {}".format(peer.type, peer.host, peer.port, reason.getErrorMessage()))
