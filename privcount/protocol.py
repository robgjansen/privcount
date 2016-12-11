import logging, json, math

from time import time
from os import _exit, urandom, path
from base64 import b64encode, b64decode

from twisted.internet import reactor
from twisted.protocols.basic import LineOnlyReceiver

from cryptography.hazmat.primitives.hashes import SHA256

from privcount.counter import get_events_for_counters, get_valid_events
from privcount.crypto import CryptoHash, get_hmac, verify_hmac, b64_padded_length

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
        self.privcount_role = None

        '''here we use the LineOnlyReceiver's MAX_LENGTH to drop the connection
        if we receive too much data before the handshake validates it
        the handshake process itself transfers very little, so we can get
        away with a small buffer - after the handshake suceeds, we allow lines
        of longer length'''
        self.MAX_LENGTH = 256
        try:
            # make sure the maximum line length will accommodate the longest
            # handshake line, including base64 encoded cookie and HMAC
            assert self.MAX_LENGTH >= (
                len(PrivCountProtocol.HANDSHAKE2) +
                len(PrivCountProtocol.handshake_prefix_str(
                        PrivCountProtocol.HANDSHAKE2,
                        PrivCountProtocol.ROLE_CLIENT)) +
                PrivCountProtocol.COOKIE_B64_BYTES +
                PrivCountProtocol.HMAC_B64_BYTES)
        except BaseException as e:
            # catch errors and terminate the process
            logging.error(
                "Exception {} while initialising PrivCountProtocol instance"
                .format(e))
            # die immediately using os._exit()
            # we can't use sys.exit() here, because twisted catches and logs it
            _exit(1)
        except:
            # catch exceptions that don't derive from BaseException
            logging.error(
                "Unknown Exception while processing event type: {} payload: {}"
                .format(event_type, event_payload))
            _exit(1)

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
            if event_type.startswith(PrivCountProtocol.HANDSHAKE_COMMON):
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
        except:
            # catch exceptions that don't derive from BaseException
            logging.error(
                "Unknown Exception while processing event type: {} payload: {}"
                .format(event_type, event_payload))
            _exit(1)

        if not self.is_valid_connection:
            self.protocol_failed()

    # PrivCount uses a HMAC-SHA256-based handshake to verify that client and
    # server both know a shared secret key, without revealing the key itself

    # The numbers of space-seprated parts on various protocol handshake lines

    # The common prefix for every protocol handshake line:
    # HANDSHAKEN VERSION ROLE TYPE
    HANDSHAKE_PREFIX_PARTS = 4
    # ServerCookie
    HANDSHAKE1_PARTS = HANDSHAKE_PREFIX_PARTS + 1
    # ClientCookie HMAC
    HANDSHAKE2_PARTS = HANDSHAKE_PREFIX_PARTS + 2
    # HMAC
    HANDSHAKE3_PARTS = HANDSHAKE_PREFIX_PARTS + 1
    # SUCCESS
    HANDSHAKE4_PARTS = HANDSHAKE_PREFIX_PARTS + 1
    # FAIL
    HANDSHAKE_FAIL_PARTS = HANDSHAKE_PREFIX_PARTS + 1

    # The first token on each handshake line
    # These tokens should be unique, because the handshake 2 & 3 tokens are
    # used as part of the hash prefix.
    HANDSHAKE_COMMON = 'HANDSHAKE'
    HANDSHAKE1 = HANDSHAKE_COMMON + '1-SERVER-COOKIE'
    HANDSHAKE2 = HANDSHAKE_COMMON + '2-CLIENT-COOKIE-HMAC'
    HANDSHAKE3 = HANDSHAKE_COMMON + '3-SERVER-HMAC'
    assert HANDSHAKE2 != HANDSHAKE3
    HANDSHAKE4 = HANDSHAKE_COMMON + '4-CLIENT-VERIFY'

    # The common prefix tokens on each handshake line

    # The first version this handshake was introduced in
    HANDSHAKE_VERSION = 'PRIVCOUNT-020'
    # The role of the node sending the handshake
    ROLE_CLIENT = 'CLIENT'
    ROLE_SERVER = 'SERVER'
    # The hash construction used for the handshake
    assert CryptoHash == SHA256
    HANDSHAKE_TYPE = 'HMAC-SHA256'
    # The message used for HANDSHAKE 4 success
    HANDSHAKE_SUCCESS = 'SUCCESS'
    # The message is used for HANDSHAKE 2-4 failure
    HANDSHAKE_FAIL = 'FAIL'

    # The number of bytes in a cookie and a hash
    COOKIE_BYTES = CryptoHash.digest_size
    COOKIE_B64_BYTES = b64_padded_length(COOKIE_BYTES)
    HMAC_BYTES = CryptoHash.digest_size
    HMAC_B64_BYTES = b64_padded_length(HMAC_BYTES)
    SECRET_BYTES = CryptoHash.digest_size
    SECRET_B64_BYTES = b64_padded_length(SECRET_BYTES)

    # The early exits in the verification code below are susceptible to
    # timing attacks. However, these timing attacks are only an issue if
    # the timing reveals secret information, such as any bits in the long-term
    # secret key, or a significant proportion of the bits in the short-term
    # cookies

    # It may also be possible to hold a connection open indefinitely while
    # brute-forcing the secret key. If this were a feasible attack, we could
    # limit it by making the cookies expire after a few minutes.

    # Finally, while a HMAC is a proven cryptographic primitive, this
    # particular construction has not been reviewed by a cryptographer, nor
    # has this particular implementation undergone cryptographic review.
    # It likely contains at least one bug, and any bugs might affect security.

    @staticmethod
    def handshake_prefix_str(handshake_stage, sender_role):
        '''
        Return the handshake prefix for stage and role:
        stage VERSION role TYPE
        '''
        return "{} {} {} {}".format(handshake_stage,
                                    PrivCountProtocol.HANDSHAKE_VERSION,
                                    sender_role,
                                    PrivCountProtocol.HANDSHAKE_TYPE)

    @staticmethod
    def handshake_prefix_verify(handshake, handshake_stage, sender_role):
        '''
        If the prefix of handshake matches the expected prefix for stage and
        role, return True.
        Otherwise, return False.
        '''
        parts = handshake.strip().split()
        if len(parts) < PrivCountProtocol.HANDSHAKE_PREFIX_PARTS:
            logging.warning("Invalid handshake: not enough parts {} expected >= {}"
                            .format(len(parts),
                                    PrivCountProtocol.HANDSHAKE_PREFIX_PARTS))
            return False
        if parts[0] != handshake_stage:
            logging.warning("Invalid handshake: wrong stage {} expected {}"
                            .format(parts[0],
                                    handshake_stage))
            return False
        if parts[1] != PrivCountProtocol.HANDSHAKE_VERSION:
            logging.warning("Invalid handshake: wrong version {} expected {}"
                            .format(parts[1],
                                    PrivCountProtocol.HANDSHAKE_VERSION))
            return False
        if parts[2] != sender_role:
            logging.warning("Invalid handshake: wrong role {} expected {}"
                            .format(parts[2],
                                    sender_role))
            return False
        if parts[3] != PrivCountProtocol.HANDSHAKE_TYPE:
            logging.warning("Invalid handshake: wrong type {} expected {}"
                            .format(parts[3],
                                    PrivCountProtocol.HANDSHAKE_TYPE))
            return False
        return True

    @staticmethod
    def handshake_cookie_get():
        '''
        Return random cookie bytes for use in the privcount handshake.
        '''
        cookie = urandom(PrivCountProtocol.COOKIE_BYTES)
        assert PrivCountProtocol.handshake_cookie_verify(b64encode(cookie))
        return cookie

    @staticmethod
    def handshake_cookie_verify(b64_cookie):
        '''
        If b64_cookie matches the expected format for a base-64 encoded
        privcount cookie, return the decoded cookie.
        Otherwise, return False.
        Raises an exception if the cookie is not correctly padded base64.
        '''
        if len(b64_cookie) != PrivCountProtocol.COOKIE_B64_BYTES:
            logging.warning("Invalid cookie: wrong encoded length {} expected {}"
                            .format(len(b64_cookie),
                                    PrivCountProtocol.COOKIE_B64_BYTES))
            return False
        cookie = b64decode(b64_cookie)
        if len(cookie) != PrivCountProtocol.COOKIE_BYTES:
            logging.warning("Invalid cookie: wrong decoded length {} expected {}"
                            .format(len(cookie),
                                    PrivCountProtocol.COOKIE_BYTES))
            return False
        return cookie

    @staticmethod
    def handshake_secret_generate():
        '''
        Return a base-64 encoded random secret key for use in the privcount
        handshake. This secret needs to be the same on the server and clients.
        '''
        handshake_key = b64encode(urandom(PrivCountProtocol.SECRET_BYTES))
        assert PrivCountProtocol.handshake_secret_verify(handshake_key)
        return handshake_key

    @staticmethod
    def handshake_secret_load(secret_handshake_path, create=False):
        '''
        Load and decode the base64 encoded secret handshake string from the
        file at secret_handshake_path.
        If create is true, and the file does not exist, create a new file
        containing a random, base64-encoded secret handshake string.
        '''
        # generate a new key if the file does not exist
        # having the protocol deal with config files is an abstraction layer
        # violation, but it yields better security properties, as only the
        # protocol layer ever knows the secret key (and it is discarded after
        # it is used to generate or verify the HMACs)
        if not path.exists(secret_handshake_path):
            secret_handshake = PrivCountProtocol.handshake_secret_generate()
            with open(secret_handshake_path, 'w') as fin:
                fin.write(secret_handshake)
        # read from the file (even if we just generated it)
        with open(secret_handshake_path, 'r') as fin:
            # read the whole file, but ignore whitespace
            secret_handshake = fin.read().strip()
        # decode
        secret_handshake = PrivCountProtocol.handshake_secret_verify(
            secret_handshake)
        return secret_handshake

    @staticmethod
    def handshake_secret_verify(handshake_key):
        '''
        If secret matches the expected format for a base-64 encoded
        privcount secret handshake key, return the decoded secret.
        Otherwise, return False.
        Raises an exception if the secret is not correctly padded base64.
        '''
        # The secret and cookie are the same size and encoding, this just works
        assert (PrivCountProtocol.COOKIE_B64_BYTES ==
                PrivCountProtocol.SECRET_B64_BYTES)
        assert (PrivCountProtocol.COOKIE_BYTES ==
                PrivCountProtocol.SECRET_BYTES)
        return PrivCountProtocol.handshake_cookie_verify(handshake_key)


    def handshake_secret(self):
        '''
        Return the secret handshake key, using the file path configured
        by the factory.
        If the factory has no file path, return False.
        '''
        # This is an abstraction layer violation, but it's necessary
        handshake_key_path = self.factory.get_secret_handshake_path()
        if handshake_key_path is not None:
            return PrivCountProtocol.handshake_secret_load(handshake_key_path)
        else:
            return False

    @staticmethod
    def handshake_hmac_get(handshake_key, prefix, server_cookie,
                           client_cookie):
        '''
        Return HMAC(handshake_key, prefix | server_cookie | client_cookie),
        base-64 encoded.
        '''
        hmac = b64encode(get_hmac(handshake_key,
                                  prefix,
                                  server_cookie +
                                  client_cookie))
        assert PrivCountProtocol.handshake_hmac_verify(hmac,
                                                       handshake_key,
                                                       prefix,
                                                       server_cookie,
                                                       client_cookie)
        return hmac

    @staticmethod
    def handshake_hmac_decode(b64_hmac):
        '''
        If b64_hmac matches the expected format for a base-64 encoded
        privcount HMAC, return the decoded HMAC.
        Otherwise, return False.
        Raises an exception if the hmac is not correctly padded base64.
        '''
        # The HMAC and cookie are the same size and encoding, this just works
        assert (PrivCountProtocol.COOKIE_B64_BYTES ==
                PrivCountProtocol.HMAC_B64_BYTES)
        assert (PrivCountProtocol.COOKIE_BYTES == PrivCountProtocol.HMAC_BYTES)
        return PrivCountProtocol.handshake_cookie_verify(b64_hmac)

    @staticmethod
    def handshake_hmac_verify(b64_hmac, handshake_key, prefix, server_cookie,
                              client_cookie):
        '''
        If b64_hmac matches the expected format for a base-64 encoded
        privcount HMAC, and the HMAC matches the expected HMAC for
        handshake_key, prefix, and the cookies, return True.
        Otherwise, return False.
        Raises an exception if the HMAC is not correctly padded base64.
        '''
        hmac = PrivCountProtocol.handshake_hmac_decode(b64_hmac)
        if not hmac:
            logging.warning("Invalid hmac: wrong format")
            return False
        if not verify_hmac(hmac,
                           handshake_key,
                           prefix,
                           server_cookie +
                           client_cookie):
            logging.warning("Invalid hmac: verification failed")
            return False
        return True

    def handshake1_str(self):
        '''
        Return a string for server handshake stage 1:
        HANDSHAKE1 VERSION SERVER TYPE ServerCookie
        '''
        assert self.privcount_role == PrivCountProtocol.ROLE_SERVER
        prefix = self.handshake_prefix_str(PrivCountProtocol.HANDSHAKE1,
                                           self.privcount_role)
        h1 = "{} {}".format(prefix,
                            b64encode(self.server_cookie))
        logging.debug("Sent handshake: {}".format(h1))
        assert self.handshake1_verify(h1)
        return h1

    @staticmethod
    def handshake1_verify(handshake):
        '''
        If handshake matches the expected format for HANDSHAKE1,
        return the server cookie.
        Otherwise, return False.
        Raises an exception if the server cookie is not correctly padded
        base64.
        '''
        if not PrivCountProtocol.handshake_prefix_verify(
                                     handshake,
                                     PrivCountProtocol.HANDSHAKE1,
                                     PrivCountProtocol.ROLE_SERVER):
            return False
        parts = handshake.strip().split()
        if len(parts) != PrivCountProtocol.HANDSHAKE1_PARTS:
            logging.warning("Invalid handshake: wrong number of parts {} expected {}"
                            .format(len(parts),
                                    PrivCountProtocol.HANDSHAKE1_PARTS))
            return False
        server_cookie = PrivCountProtocol.handshake_cookie_verify(
            parts[PrivCountProtocol.HANDSHAKE_PREFIX_PARTS])
        return server_cookie

    def handshake2_str(self):
        '''
        Return a string for client handshake stage 2:
        HANDSHAKE2 VERSION CLIENT TYPE ClientCookie
        HMAC(Key, HandshakePrefix2 | ServerCookie | ClientCookie)
        '''
        assert self.privcount_role == PrivCountProtocol.ROLE_CLIENT
        prefix = self.handshake_prefix_str(PrivCountProtocol.HANDSHAKE2,
                                           self.privcount_role)
        h2 = "{} {} {}".format(prefix,
                               b64encode(self.client_cookie),
                               self.handshake_hmac_get(self.handshake_secret(),
                                                       prefix,
                                                       self.server_cookie,
                                                       self.client_cookie))
        assert self.handshake2_verify(h2,
                                      self.handshake_secret(),
                                      self.server_cookie)
        logging.debug("Sent handshake: {}".format(h2))
        return h2

    @staticmethod
    def handshake2_verify(handshake, handshake_key, server_cookie):
        '''
        If handshake matches the expected format for HANDSHAKE2,
        and the HMAC verifies using handshake_key and server cookie,
        and the client cookie does not match the server cookie,
        return the client cookie.
        Otherwise, return False.
        Raises an exception if the client cookie or the HMAC are not
        correctly padded base64.
        '''
        if not PrivCountProtocol.handshake_prefix_verify(
                                     handshake,
                                     PrivCountProtocol.HANDSHAKE2,
                                     PrivCountProtocol.ROLE_CLIENT):
            return False
        parts = handshake.strip().split()
        if len(parts) != PrivCountProtocol.HANDSHAKE2_PARTS:
            logging.warning("Invalid handshake: wrong number of parts {} expected {}"
                            .format(len(parts),
                                    PrivCountProtocol.HANDSHAKE2_PARTS))
            return False
        client_cookie = PrivCountProtocol.handshake_cookie_verify(
            parts[PrivCountProtocol.HANDSHAKE_PREFIX_PARTS])
        if not client_cookie:
            return False
        # choosing the same cookie is extremely unlikely, unless the client
        # just re-used the server cookie
        # The hadshake is immune from this kind of attack by construction:
        #  - the client and server HMACs use a distinct prefix
        #  - the server provides its cookie before its HMAC, so it can't
        #    copy the client cookie and HMAC
        #  - the client provides its HMAC before the server HMAC, so it can't
        #    copy the server cookie and HMAC
        # but we do the check anyway, because it's just weird if this happens
        if client_cookie == server_cookie:
            logging.warning("Invalid handshake: client cookie matches server cookie")
            return False
        hmac = parts[PrivCountProtocol.HANDSHAKE_PREFIX_PARTS + 1]
        prefix = PrivCountProtocol.handshake_prefix_str(
                                       PrivCountProtocol.HANDSHAKE2,
                                       PrivCountProtocol.ROLE_CLIENT)
        if not PrivCountProtocol.handshake_hmac_verify(hmac,
                                                       handshake_key,
                                                       prefix,
                                                       server_cookie,
                                                       client_cookie):
            return False
        return client_cookie

    def handshake3_str(self):
        '''
        Return a string for server handshake stage 3:
        HANDSHAKE3 VERSION SERVER TYPE
        HMAC(Key, HandshakePrefix3 | ServerCookie | ClientCookie)
        '''
        assert self.privcount_role == PrivCountProtocol.ROLE_SERVER
        prefix = self.handshake_prefix_str(PrivCountProtocol.HANDSHAKE3,
                                           self.privcount_role)
        h3 = "{} {}".format(prefix,
                            self.handshake_hmac_get(self.handshake_secret(),
                                                    prefix,
                                                    self.server_cookie,
                                                    self.client_cookie))
        assert self.handshake3_verify(h3,
                                      self.handshake_secret(),
                                      self.server_cookie,
                                      self.client_cookie)
        logging.debug("Sent handshake: {}".format(h3))
        return h3

    @staticmethod
    def handshake3_verify(handshake, handshake_key, server_cookie,
                          client_cookie):
        '''
        If handshake matches the expected format for HANDSHAKE3,
        and the HMAC verifies, return True.
        Otherwise, return False.
        Raises an exception if the HMAC is not correctly padded base64.
        '''
        if not PrivCountProtocol.handshake_prefix_verify(
                                     handshake,
                                     PrivCountProtocol.HANDSHAKE3,
                                     PrivCountProtocol.ROLE_SERVER):
            return False
        parts = handshake.strip().split()
        if len(parts) != PrivCountProtocol.HANDSHAKE3_PARTS:
            logging.warning("Invalid handshake: wrong number of parts {} expected {}"
                            .format(len(parts),
                                    PrivCountProtocol.HANDSHAKE3_PARTS))
            return False
        hmac = parts[PrivCountProtocol.HANDSHAKE_PREFIX_PARTS]
        prefix = PrivCountProtocol.handshake_prefix_str(
                                       PrivCountProtocol.HANDSHAKE3,
                                       PrivCountProtocol.ROLE_SERVER)
        if not PrivCountProtocol.handshake_hmac_verify(hmac,
                                                       handshake_key,
                                                       prefix,
                                                       server_cookie,
                                                       client_cookie):
            return False
        return True

    def handshake4_str(self):
        '''
        Return a string for client handshake stage 4:
        HANDSHAKE4 VERSION CLIENT TYPE SUCCESS
        '''
        assert self.privcount_role == PrivCountProtocol.ROLE_CLIENT
        prefix = self.handshake_prefix_str(PrivCountProtocol.HANDSHAKE4,
                                           self.privcount_role)
        h4 = "{} {}".format(prefix,
                            PrivCountProtocol.HANDSHAKE_SUCCESS)
        assert self.handshake4_verify(h4)
        logging.debug("Sent handshake: {}".format(h4))
        return h4

    @staticmethod
    def handshake4_verify(handshake):
        '''
        If handshake matches the expected format for HANDSHAKE4,
        and the message is SUCCESS, return True.
        Otherwise, return False.
        '''
        if not PrivCountProtocol.handshake_prefix_verify(
                                     handshake,
                                     PrivCountProtocol.HANDSHAKE4,
                                     PrivCountProtocol.ROLE_CLIENT):
            return False
        parts = handshake.strip().split()
        if len(parts) != PrivCountProtocol.HANDSHAKE4_PARTS:
            logging.warning("Invalid handshake: wrong number of parts {} expected {}"
                            .format(len(parts),
                                    PrivCountProtocol.HANDSHAKE4_PARTS))
            return False
        message = parts[PrivCountProtocol.HANDSHAKE_PREFIX_PARTS]
        if message != PrivCountProtocol.HANDSHAKE_SUCCESS:
            logging.warning("Invalid handshake: message was not SUCCESS")
            return False
        return True

    def handshake_fail_str(self, handshake_stage):
        '''
        Return a string for handshake stage failure:
        HANDSHAKEN VERSION ROLE TYPE FAIL
        '''
        prefix = self.handshake_prefix_str(handshake_stage,
                                           self.privcount_role)
        hf = "{} {}".format(prefix,
                            PrivCountProtocol.HANDSHAKE_FAIL)
        # A failed handshake should not verify as any other handshake type
        # (we don't expect a failed handshake1, but check anyway)
        assert not self.handshake1_verify(hf)
        # use the real key if we have it
        handshake_k = self.handshake_secret()
        if not handshake_k:
            handshake_k = self.handshake_secret_generate()
        # use the real cookies if we have them
        server_c = self.server_cookie
        if server_c is not None:
            server_c = self.handshake_cookie_get()
        client_c = self.client_cookie
        if client_c is not None:
            client_c = self.handshake_cookie_get()
        assert not self.handshake2_verify(hf, handshake_k, server_c)
        assert not self.handshake3_verify(hf, handshake_k, server_c, client_c)
        assert not self.handshake4_verify(hf)
        # Check that it verifies as a correctly formatted failure message
        assert self.handshake_fail_verify(hf)
        logging.debug("Sent handshake: {}".format(hf))
        return hf

    @staticmethod
    def handshake_fail_verify(handshake):
        '''
        If handshake matches the expected format for a failed handshake at
        any stage, return True.
        Otherwise, return False.

        Usage note:
        If a handshake does not match any expected format (including the
        fail format), it should be considered a failure.
        '''
        # Handshakes 2-4 can contain failure responses
        # These calls lead to spurious log messages
        if (not PrivCountProtocol.handshake_prefix_verify(
                                      handshake,
                                      PrivCountProtocol.HANDSHAKE2,
                                      PrivCountProtocol.ROLE_CLIENT) and
            not PrivCountProtocol.handshake_prefix_verify(
                                      handshake,
                                      PrivCountProtocol.HANDSHAKE3,
                                      PrivCountProtocol.ROLE_SERVER) and
            not PrivCountProtocol.handshake_prefix_verify(
                                      handshake,
                                      PrivCountProtocol.HANDSHAKE4,
                                      PrivCountProtocol.ROLE_CLIENT)):
            logging.warning("Invalid handshake: failure message did not match any possible handshake format")
            return False
        parts = handshake.strip().split()
        # A handshake fail consists of the prefix and a fail message
        if len(parts) != PrivCountProtocol.HANDSHAKE_FAIL_PARTS:
            logging.warning("Invalid handshake: wrong number of parts {} expected {}"
                            .format(len(parts),
                                    PrivCountProtocol.HANDSHAKE_FAIL_PARTS))
            return False
        message = parts[PrivCountProtocol.HANDSHAKE_PREFIX_PARTS]
        if message != PrivCountProtocol.HANDSHAKE_FAIL:
            logging.warning("Invalid handshake: message was not FAIL")
            return False
        return True

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
        self.privcount_role = PrivCountProtocol.ROLE_SERVER
        self.last_sent_time = 0.0
        self.client_uid = None

    def connectionMade(self): # overrides twisted function
        PrivCountProtocol.connectionMade(self)
        self.send_handshake_event()

    def send_handshake_event(self):
        '''
        initiate the handshake with the client
        '''
        self.server_cookie = self.handshake_cookie_get()
        self.sendLine(self.handshake1_str())

    def handle_handshake_event(self, event_type, event_payload):
        '''
        If the received handshake is valid, send the next handshake in the
        sequence. Otherwise, fail the handshaking process.
        '''
        # reconstitute the event line
        event_line = event_type + ' ' + event_payload
        logging.debug("Received handshake: {}".format(event_line))

        if event_type == PrivCountProtocol.HANDSHAKE2:
            self.client_cookie = self.handshake2_verify(
                event_line,
                self.handshake_secret(),
                self.server_cookie)
            if self.client_cookie:
                self.sendLine(self.handshake3_str())
            else:
                self.client_cookie = None
                # We don't need to securely delete this cookie, because it
                # is single-use, and used for authentication only
                self.server_cookie = None
                self.sendLine(self.handshake_fail_str(
                                       PrivCountProtocol.HANDSHAKE3))
                self.handshake_failed()
        elif event_type == PrivCountProtocol.HANDSHAKE4:
            if self.handshake4_verify(event_line):
                self.handshake_succeeded()
            else:
                self.handshake_failed()
        else:
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

    def __init__(self, factory):
        PrivCountProtocol.__init__(self, factory)
        self.privcount_role = PrivCountProtocol.ROLE_CLIENT

    def handshake_succeeded(self):
        PrivCountProtocol.handshake_succeeded(self)
        # for a reconnecting client, reset the exp backoff delay
        self.factory.resetDelay()

    def protocol_succeeded(self):
        PrivCountProtocol.protocol_succeeded(self)
        # for a reconnecting client, don't reconnect after this disconnection
        self.factory.stopTrying()

    def handle_handshake_event(self, event_type, event_payload):
        '''
        If the received handshake is valid, send the next handshake in the
        sequence. Otherwise, fail the handshaking process.
        '''
        # reconstitute the event line
        event_line = event_type + ' ' + event_payload
        logging.debug("Received handshake: {}".format(event_line))

        if event_type == PrivCountProtocol.HANDSHAKE1:
            self.server_cookie = self.handshake1_verify(event_line)
            if self.server_cookie:
                self.client_cookie = self.handshake_cookie_get()
                self.sendLine(self.handshake2_str())
            else:
                self.server_cookie = None
                self.sendLine(self.handshake_fail_str(
                                       PrivCountProtocol.HANDSHAKE2))
                self.handshake_failed()
        elif event_type == PrivCountProtocol.HANDSHAKE3:
            if self.handshake3_verify(event_line,
                                      self.handshake_secret(),
                                      self.server_cookie,
                                      self.client_cookie):
                self.sendLine(self.handshake4_str())
                self.handshake_succeeded()
            else:
                self.sendLine(self.handshake_fail_str(
                                       PrivCountProtocol.HANDSHAKE4))
                self.handshake_failed()
        else:
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
        self.collection_events = None
        self.active_events = None

    def connectionMade(self):
        '''
        Initiate authentication with the control port, and put the protocol in
        the 'authenticating' state.
        Overrides twisted function.
        '''
        peer = self.transport.getPeer()
        logging.debug("Connection with {}:{}:{} was made".format(peer.type, peer.host, peer.port))
        # we must authenticate first, before doing anything else
        self.sendLine("AUTHENTICATE")
        self.state = 'authenticating'

    def startCollection(self, counter_list, event_list=None):
        '''
        Enable events for all the events required by counter_list, and all
        events explictly specified in event_list. After every successful
        control port connection, re-enable the events.
        '''
        self.collection_events = set()
        if counter_list is not None:
            self.collection_events |= get_events_for_counters(counter_list)
        if event_list is not None:
            for event in event_list:
                upper_event = event.upper()
                if upper_event in get_valid_events():
                    self.collection_events.add(upper_event)
                else:
                    logging.warning("Ignored unknown event: {}".format(event))
        logging.warning("Starting PrivCount collection with events: {} from counters: {} and events: {}"
                        .format(" ".join(self.collection_events),
                                "(none)" if counter_list is None else
                                " ".join(counter_list),
                                "(none)" if event_list is None else
                                " ".join(event_list)))
        self.enableEvents()

    def stopCollection(self):
        '''
        Disable all events. Remain connected to the control port, but wait for
        the next collection to start.
        '''
        self.collection_events = None
        self.disableEvents()
        # let the user know that we're waiting
        logging.warning("Waiting for PrivCount collection to start")

    def enableEvents(self):
        '''
        If we are ready to handle events, and we know which events we want
        to process, turn privcount internals on, and turn on the events for
        the stored list of counters.
        '''
        # only start PrivCount if we are ready to handle events,
        # and know which events we want to handle
        if self.collection_events is None:
            return
        if self.state is None:
            return
        if self.state == 'waiting' or self.state == 'processing':
            self.active_events = self.collection_events
            self.sendLine("SETCONF EnablePrivCount=1")
            self.sendLine("SETEVENTS {}".format(" ".join(self.active_events)))
            self.state = 'processing'
            logging.info("Enabled PrivCount events: {}"
                         .format(" ".join(self.active_events)))

    def disableEvents(self):
        '''
        Turn privcount events and privcount internals off
        '''
        # try to turn events off regardless of the state, as long as we have
        # connected and authenticated, this will work
        # (and if it doesn't, we haven't lost anything)
        if self.state is not None:
            if self.active_events is not None:
                logging.info("Disabled PrivCount events: {}"
                             .format(" ".join(self.active_events)))
            self.sendLine("SETCONF EnablePrivCount=0")
            self.sendLine("SETEVENTS")
            self.active_events = None
            self.state = 'waiting'

    def setDiscoveredValue(self, function_name, value, value_name, peer):
        '''
        When we discover value from the relay, call set_function to set it,
        and log a message containing value_name if this fails.
        '''
        try:
            # Equivalent to self.factory.set_*(value)
            if not getattr(self.factory, function_name)(value):
                logging.warning("Connection with {}:{}:{}: bad {}: {}"
                                .format(peer.type, peer.host, peer.port,
                                        value_name, value))
        except AttributeError as e:
            logging.warning("Connection with {}:{}:{}: sent {}: {}, but factory raised {}"
                            .format(peer.type, peer.host, peer.port,
                                    value_name, value, e))

    def lineReceived(self, line):
        '''
        Check that authentication was successful.
        If so, put the protocol in the 'discovering' state, and ask the relay
        for information about itself.
        When the fingerprint is receved, put the protocol in the 'waiting'
        state.
        When the round is started, put the protocol in the 'processing' state,
        and send the list of events we want.
        When events are received, process them.
        Overrides twisted function.
        '''
        peer = self.transport.getPeer()
        line = line.strip()
        logging.debug("Received line '{}' from {}:{}:{}".format(line, peer.type, peer.host, peer.port))

        if self.state == 'authenticating' and line == "250 OK":
            # Turn off privcount events, so we don't get spurious responses
            self.disableEvents()
            # Just ask for all the info all at once
            self.sendLine("GETCONF Nickname")
            self.sendLine("GETCONF ORPort")
            self.sendLine("GETCONF DirPort")
            self.sendLine("GETINFO version")
            self.sendLine("GETINFO address")
            self.sendLine("GETINFO fingerprint")
            self.state = 'discovering'
        elif self.state == 'discovering':
            # -- These cases continue discovering, or quit() -- #
            # It doesn't have PrivCount
            if line.startswith("552 Unrecognized option"):
                logging.critical("Connection with {}:{}:{}: does not support PrivCount".format(peer.type, peer.host, peer.port))
                self.quit()
            # It's a relay, and it's just told us its Nickname
            elif line.startswith("250 Nickname="):
                _, _, nickname = line.partition("Nickname=") # returns: part1, separator, part2
                self.setDiscoveredValue('set_nickname', nickname, 'Nickname',
                                        peer)
            # It doesn't have a Nickname, maybe it's a client?
            # But we'll catch that when we check the fingerprint, so just ignore this response
            elif line == "250 Nickname":
                logging.info("Connection with {}:{}:{}: no Nickname".format(peer.type, peer.host, peer.port))
            # It's a relay, and it's just told us its ORPort
            elif line.startswith("250 ORPort="):
                _, _, orport = line.partition("ORPort=") # returns: part1, separator, part2
                self.setDiscoveredValue('set_orport', orport, 'ORPort', peer)
            # It doesn't have an ORPort, maybe it's a client?
            # But we'll catch that when we check the fingerprint, so just ignore this response
            elif line == "250 ORPort":
                logging.warning("Connection with {}:{}:{}: no ORPort".format(peer.type, peer.host, peer.port))
            # It's a relay, and it's just told us its DirPort
            elif line.startswith("250 DirPort="):
                _, _, dirport = line.partition("DirPort=") # returns: part1, separator, part2
                self.setDiscoveredValue('set_dirport', dirport, 'DirPort',
                                        peer)
            elif line == "250 DirPort":
                logging.info("Connection with {}:{}:{}: no DirPort".format(peer.type, peer.host, peer.port))
            # It's just told us its version
            # The control spec assumes that Tor always has a version, so there's no error case
            elif line.startswith("250-version="):
                _, _, version = line.partition("version=") # returns: part1, separator, part2
                self.setDiscoveredValue('set_version', version, 'version',
                                        peer)
            # It's just told us its address
            elif line.startswith("250-address="):
                _, _, address = line.partition("address=") # returns: part1, separator, part2
                self.setDiscoveredValue('set_address', address, 'address',
                                        peer)
            # We asked for its address, and it couldn't find it. That's weird.
            elif line == "551 Address unknown":
                logging.info("Connection with {}:{}:{}: does not know its own address".format(peer.type, peer.host, peer.port))
            # -- fingerprint discovery ends in waiting or quit() --
            # It's a relay, and it's just told us its fingerprint
            elif line.startswith("250-fingerprint="):
                _, _, fingerprint = line.partition("fingerprint=") # returns: part1, separator, part2
                self.setDiscoveredValue('set_fingerprint', fingerprint,
                                        'fingerprint', peer)
                # waiting mode will skip all further lines, until collection
                self.state = 'waiting'
            # We asked for its fingerprint, and it said it's a client
            elif line == "551 Not running in server mode":
                logging.warning("Connection with {}:{}:{} failed: not a relay".format(peer.type, peer.host, peer.port))
                self.quit()
            else:
                self.handleUnexpectedLine(line, peer)

            # If we already know what events we should have enabled, start
            # processing them
            if self.state == 'waiting':
                self.enableEvents()
        elif self.state == 'waiting' and line.startswith("2"):
            # log ok events while we're waiting for round start
            self.handleUnexpectedLine(line, peer)
        elif self.state == 'processing' and line.startswith("650 PRIVCOUNT_"):
            parts = line.split(" ")
            assert len(parts) > 1
            # skip unwanted events
            if not parts[1] in self.active_events:
                if not parts[1] in get_valid_events():
                    logging.warning("Unknown event type {}".format(line))
                else:
                    logging.warning("Unwanted event type {}".format(line))
            # skip empty events
            elif len(parts) <= 2:
                # send the event, including the event type
                logging.warning("Event with no data {}".format(line))
            elif not self.factory.handle_event(parts[1:]):
                self.quit()
        else:
            self.handleUnexpectedLine(line, peer)

    def handleUnexpectedLine(self, line, peer):
        '''
        Log any unexpected responses at an appropriate level.
        Quit on error responses.
        '''
        if line == "250 OK":
            logging.debug("Connection with {}:{}:{}: ok response: '{}'".format(peer.type, peer.host, peer.port, line))
        elif line.startswith("650 PRIVCOUNT_"):
            logging.warning("Connection with {}:{}:{}: unexpected event: '{}'".format(peer.type, peer.host, peer.port, line))
        elif line.startswith("5"):
            logging.warning("Connection with {}:{}:{}: unexpected response: '{}'".format(peer.type, peer.host, peer.port, line))
            self.quit()
        elif line.startswith("2"):
            logging.info("Connection with {}:{}:{}: ok response: '{}'".format(peer.type, peer.host, peer.port, line))
        else:
            logging.warning("Connection with {}:{}:{}: unexpected response: '{}'".format(peer.type, peer.host, peer.port, line))
            self.quit()

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
    SETCONF EnablePrivCount=1
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
    SETEVENTS PRIVCOUNT_STREAM_BYTES_TRANSFERRED PRIVCOUNT_STREAM_ENDED
    552 Unrecognized event "PRIVCOUNT_STREAM_BYTES_TRANSFERRED"
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

        # We use " quotes, but tor uses ' quotes. This should not matter.
        if not self.authenticated:
            if line == "AUTHENTICATE":
                self.sendLine("250 OK")
                self.authenticated = True
            else:
                self.sendLine("514 Authentication required.")
                self.transport.loseConnection()
        elif len(parts) > 0:
            if parts[0] == "SETEVENTS":
                # events are case-insensitive, so convert to uppercase
                upper_setevents = map(str.upper, parts[1:])
                # already uppercase
                upper_known_events = get_valid_events()
                # if every requested event is in the known events
                # if there are no requested events, that's ok, it turns events
                # off
                if set(upper_setevents).issubset(upper_known_events):
                    self.sendLine("250 OK")
                    self.factory.start_injecting()
                else:
                    unknown_events = set(upper_setevents).difference(
                        upper_known_events)
                    assert len(unknown_events) > 0
                    # this line displays the event name uppercased
                    # that's a minor, irrelevant protocol difference
                    self.sendLine('552 Unrecognized event "{}"'
                                  .format(unknown_events[0]))
                    self.factory.stop_injecting()
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
            elif parts[0] == "SETCONF":
                # the correct response to an empty SETCONF is an empty OK
                if len(parts) == 1:
                    self.sendLine("250 OK")
                # Like GETCONF, SETCONF is case-insensitive, and returns one
                # line: "250 OK"
                elif len(parts) == 2 and parts[1].lower().startswith("enableprivcount"):
                    # just ignore the value
                    self.sendLine("250 OK")
                else:
                    # Like GETINFO, our GETCONF does not accept multiple words
                    # and it doesn't bother to strip the =
                    self.sendLine('552 Unrecognized option: Unknown option "{}". Failing.'.format(parts[1]))
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
