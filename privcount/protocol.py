# See LICENSE for licensing information

import logging, json, math, subprocess, sys, os

from time import time
from os import urandom, path
from base64 import b64encode, b64decode
from binascii import hexlify, unhexlify

from twisted.internet import task
from twisted.protocols.basic import LineOnlyReceiver

from cryptography.hazmat.primitives.hashes import SHA256

from privcount.connection import transport_info, transport_remote_info, transport_local_info
from privcount.counter import get_events_for_counters, get_valid_events
from privcount.crypto import CryptoHash, get_hmac, verify_hmac, b64_padded_length
from privcount.log import log_error, errorCallback, stop_reactor, summarise_string

PRIVCOUNT_SHORT_VERSION_STRING = '1.1.0'

def get_privcount_version():
    '''
    Return a string describing the PrivCount version
    '''
    return "{} (protocol {}, git-{})".format(
                                          PRIVCOUNT_SHORT_VERSION_STRING,
                                          PrivCountProtocol.HANDSHAKE_VERSION,
                                          privcount_git_revision())

PRIVCOUNT_GIT_CACHE = None

def privcount_git_revision():
    '''
    Return a string containing the current git revision.
    (The commit hash does not include any uncommitted working tree changes.)
    Returns None if git is not installed, or the command fails in some other
    way.
    '''
    global PRIVCOUNT_GIT_CACHE
    if PRIVCOUNT_GIT_CACHE is not None:
        return PRIVCOUNT_GIT_CACHE

    # this is exactly what tor uses to report its revision
    git_command_line = ['git', 'rev-parse', '--short=16', 'HEAD']
    try:
        PRIVCOUNT_GIT_CACHE = subprocess.check_output(git_command_line).strip()
    except subprocess.CalledProcessError as e:
        logging.info('Git revision check {} returned {} cmd "{}" output "{}"'
                     .format(git_command_line,
                             e.returncode, e.cmd, e.output))
        PRIVCOUNT_GIT_CACHE = "(no revision)"
    # if any error happens here, log but ignore it
    except BaseException as e:
        logging.info('Git revision check {} exception: "{}"'
                     .format(git_command_line, e))
        PRIVCOUNT_GIT_CACHE = "(no revision)"

    return PRIVCOUNT_GIT_CACHE

class PrivCountProtocol(LineOnlyReceiver):
    '''
    The base protocol class for PrivCount. This class logs basic connection
    information when connections are made and lost, and tracks the validity of
    connections during the handshake process and execution of the protocol.
    '''

    def __init__(self, factory):
        self.factory = factory
        self.privcount_role = None
        self.clear()
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
        # if any error happens here, die
        except BaseException as e:
            # catch errors and terminate the process
            logging.error(
                "Exception {} while initialising PrivCountProtocol instance"
                .format(e))
            log_error()

            stop_reactor(1)

    def clear(self):
        '''
        Clear all the instance variables
        '''
        self.is_valid_connection = False
        self.client_cookie = None
        self.server_cookie = None

        '''here we use the LineOnlyReceiver's MAX_LENGTH to drop the connection
        if we receive too much data before the handshake validates it
        the handshake process itself transfers very little, so we can get
        away with a small buffer - after the handshake suceeds, we allow lines
        of longer length'''
        self.MAX_LENGTH = 200

    def get_warn_length(self, is_line_received):
        '''
        Returns the line length at which we should warn, based on whether
        is_line_received or not
        '''
        if not self.is_valid_connection:
            # The handshake sends predictable line lengths, so we can adopt a
            # tight upper bound. We never warn for unvalidated received data:
            # it might be a port scanner
            return (self.MAX_LENGTH * 4) / 5
        else:
            # The PrivCount protcol sends lines that are all about the same
            # length, because blinded values are almost always the same length
            # (but the line length also increases for each extra node)
            return self.MAX_LENGTH / 2

    def check_line_length(self, line, is_line_received, is_length_exceeded):
        '''
        Warns on over-length lines, based on whether the line is received,
        and whether the line was delivered via lineLengthExceeded or not
        (sometimes, twisted's lineLengthExceeded only delivers a partial line,
        https://twistedmatrix.com/trac/ticket/6558
        and it had issues counting end of line characters
        https://twistedmatrix.com/trac/ticket/6536
        )
        Terminates the protocol or reactor if the line is over-length.
        '''
        is_length_exceeded = is_length_exceeded or len(line) > self.MAX_LENGTH
        is_unsafe_length = is_length_exceeded or len(line) > self.get_warn_length(is_line_received)
        # we trust input we send, and input from validated peers
        # don't ever warn about port scanners
        is_something_we_control = not is_line_received or self.is_valid_connection
        # if we are over the safe length, warn
        if is_unsafe_length and is_something_we_control:
            logging.warning("{} line of length {} exceeded {} of {}, {} {} connection to {}"
                            .format("Received" if is_line_received else "Generated",
                                    len(line),
                                    "MAX_LENGTH" if is_length_exceeded else "safe length",
                                    self.get_warn_length(is_line_received),
                                    "dropping" if is_length_exceeded and is_line_received else "keeping",
                                    "validated" if self.is_valid_connection else "unvalidated",
                                    transport_info(self.transport)))
        # if we send or receive an overlength line, this is a protocol error
        if is_length_exceeded:
            self.protocol_failed()
        # if we generate an overlength line, it is a coding or config bug: fail
        if is_length_exceeded and not is_line_received:
            stop_reactor(1)

    def connectionMade(self):
        '''
        overrides twisted function
        '''
        logging.debug("Connection with {} was made"
                      .format(transport_info(self.transport)))

    def lineReceived(self, line):
        '''
        overrides twisted function
        '''
        logging.debug("Received line '{}' from {}"
                      .format(line, transport_info(self.transport)))
        self.check_line_length(line, True, False)
        parts = [part.strip() for part in line.split(' ', 1)]
        if len(parts) > 0:
            event_type = parts[0]
            event_payload = parts[1] if len(parts) > 1 else ''
            self.process_event(event_type, event_payload)

    def sendLine(self, line):
        '''
        overrides twisted function
        '''
        logging.debug("Sending line '{}' to {}"
                      .format(line, transport_info(self.transport)))
        self.check_line_length(line, False, False)
        return LineOnlyReceiver.sendLine(self, line)

    def lineLengthExceeded(self, line):
        '''
        overrides twisted function
        '''
        self.check_line_length(line, True, True)
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
                .format(e, event_type, summarise_string(event_payload, 100)))
            logging.debug(
                "Exception {} while processing event type: {} payload: (full) {}"
                .format(e, event_type, event_payload))
            log_error()

            try:
                self.protocol_failed()
            except BaseException as e:
                logging.error("Exception {} in protocol failure after event type: {} payload: {}"
                              .format(e, event_type,
                                      summarise_string(event_payload, 100)))
                logging.debug("Exception {} in protocol failure after event type: {} payload: (full) {}"
                              .format(e, event_type, event_payload))

                log_error()

            # That said, terminating on exception is a denial of service
            # risk: if an untrusted party can cause an exception, they can
            # bring down the privcount network.
            # Since we ignore unrecognised events, the client would have to be
            # maliciously crafted, not just a port scanner.
            stop_reactor(1)

        if not self.is_valid_connection:
            self.protocol_failed()

    # PrivCount uses a HMAC-SHA256-based handshake to verify that client and
    # server both know a shared secret key, without revealing the key itself
    # The handshake's construction is similar to the Tor Control Port's
    # SAFECOOKIE authentication method.

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

    # The current version of the PrivCount protocol
    # Should only be updated for incompatible API changes
    # These are major version changes, according to http://semver.org/
    HANDSHAKE_VERSION = 'PRIVCOUNT-100'
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
        '''
        Called when the PrivCount handshake succeeds
        '''
        logging.debug("Handshake with {} was successful"
                      .format(transport_info(self.transport)))
        self.is_valid_connection = True
        # now allow longer lines
        # PrivCount 1.0.0 reached 6 MB with all counters, a large traffic
        # model, 20 DCs, and 10 SKs
        self.MAX_LENGTH = 30*1024*1024

    def handshake_failed(self):
        '''
        Called when the PrivCount handshake fails
        '''
        logging.warning("Handshake with {} failed"
                        .format(transport_info(self.transport)))
        self.transport.loseConnection()
        self.clear()

    def protocol_succeeded(self):
        '''
        Called when the protocol has finished successfully
        '''
        logging.debug("Protocol with {} was successful"
                      .format(transport_info(self.transport)))
        self.transport.loseConnection()
        self.clear()

    def protocol_failed(self):
        '''
        Called when the prococol finishes in failure
        '''
        # Don't log a warning, because port scanners fail the protocol
        logging.info("Protocol with {} failed"
                     .format(transport_info(self.transport)))
        self.transport.loseConnection()
        self.clear()

    def connectionLost(self, reason):
        '''
        overrides twisted function
        '''
        logging.debug("Protocol connection with {} was lost: {}"
                      .format(transport_info(self.transport),
                              reason.getErrorMessage()))
        self.clear()

    def clientConnectionFailed(self, connector, reason):
        '''
        overrides twisted function: only for clients
        '''
        logging.warning("Protocol client connection with transport: {} destination: {} failed: {}"
                        .format(transport_info(self.transport),
                                connector.getDestination(),
                                reason.getErrorMessage()))
        connector.stopConnecting()
        self.clear()

    def clientConnectionLost(self, connector, reason):
        '''
        overrides twisted function: only for clients
        '''
        logging.warning("Protocol client connection with transport: {} destination: {} was lost: {}"
                        .format(transport_info(self.transport),
                                connector.getDestination(),
                                reason.getErrorMessage()))
        connector.stopConnecting()
        self.clear()

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
        self.clear()

    def clear(self):
        '''
        Clear all the instance variables
        '''
        # this does a double-clear on init, that's ok
        PrivCountProtocol.clear(self)
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
            local = transport_local_info(self.transport)
            if local is not None:
                client_status['tally_server_address'] = local
            remote = transport_remote_info(self.transport)
            if remote is not None:
                client_status['client_address'] = remote
            client_status['clock_skew'] = 0.0
            client_status['rtt'] = 0.0

            if self.last_sent_time > 0:
                client_status['rtt'] = client_status['alive'] - self.last_sent_time
                latency = client_status['rtt'] / 2.0
                client_time = float(parts[0])
                client_status['clock_skew'] = abs(time() - latency - client_time)
                self.last_sent_time = 0
            # Share Keepers use their public key hash as their name
            # Data Collectors must each have a unique name
            # We used to disambiguate data collectors by IP address, using:
            # transport_remote_hostname(self.transport)
            self.client_uid = client_status['name']
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
                # we have to store the checkin task in the factory,
                # because the protocol is re-created on every connection
                checkin_task = self.factory.get_checkin_task()
                if checkin_task is not None and checkin_task.running:
                    checkin_task.stop()
                    self.factory.set_checkin_task(None)
                checkin_task = task.LoopingCall(self.factory.do_checkin)
                self.factory.set_checkin_task(checkin_task)
                # we ignore any errors from do_checkin, see bug #47
                checkin_deferred = checkin_task.start(period, now=False)
                checkin_deferred.addErrback(errorCallback)
                self.sendLine("CHECKIN SUCCESS")
                self.protocol_succeeded()
                return True
        return False

class TorControlProtocol(object):
    '''
    A mixin class containing common Tor Control Protocol code
    '''

    def __init__(self, factory):
        self.factory = factory
        # Events can be up to ~1kB, start warning at 2kB, reject at 20kB
        # We really don't care about the server maximum, because we only use
        # the server for testing: in production, the server is tor
        self.MAX_LENGTH = 2*10*1024
        self.clear()

    def clear(self):
        '''
        Clear all the instance variables
        '''
        self.cookie_string = None
        self.client_nonce = None
        self.server_nonce = None

    SAFECOOKIE_LENGTH = 32

    SAFECOOKIE_SERVER_NONCE_LENGTH = 32
    SAFECOOKIE_SERVER_HASH_LENGTH = 32

    # The security of SAFECOOKIE authentication does not depend on the client
    # nonce
    SAFECOOKIE_CLIENT_NONCE_MIN_VALID_LENGTH = 0
    # An arbitrary limit, 1 kilobyte is quite enough to hash
    SAFECOOKIE_CLIENT_NONCE_MAX_VALID_LENGTH = 1024
    SAFECOOKIE_CLIENT_NONCE_GENERATED_LENGTH = 32
    assert (SAFECOOKIE_CLIENT_NONCE_GENERATED_LENGTH >=
            SAFECOOKIE_CLIENT_NONCE_MIN_VALID_LENGTH)
    assert (SAFECOOKIE_CLIENT_NONCE_GENERATED_LENGTH <=
            SAFECOOKIE_CLIENT_NONCE_MAX_VALID_LENGTH)

    SAFECOOKIE_CLIENT_HASH_LENGTH = 32

    SAFECOOKIE_SERVER_HASH_KEY = \
        "Tor safe cookie authentication server-to-controller hash"
    SAFECOOKIE_CLIENT_HASH_KEY = \
        "Tor safe cookie authentication controller-to-server hash"

    # Any file shorter than this is probably a config error
    # (PrivCount does not accept zero-length passwords, even though tor does.)
    PASSWORD_MIN_VALID_LENGTH = 8
    # An arbitrary limit, 1 kilobyte is quite enough to hash
    PASSWORD_MAX_VALID_LENGTH = 1024

    @staticmethod
    def encodeControllerString(cont_str, hex_encode = True):
        '''
        Encode a string for transmission over the control port.
        If hex_encode is True, encode in hexadecimal, otherwise, encode
        in the Tor Control Protocol QuotedString format.
        Does not support the CString encoding.
        Only some strings in the tor control protocol need encoding.
        The same encodings are used by tor and the controller.
        '''
        if hex_encode:
            # hex encoded strings do not have an 0x prefix
            encoded = hexlify(cont_str)
        else: # QuotedString
            # quoted strings escape \ and " with \, then quote with "
            # the order of these replacements is important: they ensure that
            # " becomes \" rather than \\"
            cont_str = cont_str.replace("\\", "\\\\")
            cont_str = cont_str.replace("\"", "\\\"")
            encoded = "\"" + cont_str + "\""
        # sanity check
        assert TorControlProtocol.decodeControllerString(encoded) == cont_str
        return encoded

    @staticmethod
    def decodeControllerString(cont_str):
        '''
        Decode an encoded string received via the control port.
        Decodes hexadecimal, and the Tor Control Protocol QuotedString format,
        depending on the format of the input string.
        Does not support the CString encoding, or raw strings that aren't in
        one of the two supported formats.
        Throws TypeError when presented with an invalid format.
        Only some strings in the tor control protocol need encoding.
        The same encodings are used by tor and the controller.
        '''
        cont_str = cont_str.strip()
        if (cont_str.startswith("\"") and cont_str.endswith("\"") and
            len(cont_str) >= 2):
            # quoted strings escape \ and " with \, then quote with "
            # this is safe, because we check the string is "*"
            cont_str = cont_str[1:-1]
            # the order of these replacements is important: they ensure that
            # \\" becomes \" rather than "
            cont_str = cont_str.replace("\\\"", "\"")
            return cont_str.replace("\\\\", "\\")
        else:
            # assume hex, throws TypeError on invalid hex
            return unhexlify(cont_str)

    @staticmethod
    def generateNonce(len):
        '''
        Generate a nonce len bytes long.
        '''
        assert len > 0
        return urandom(len)

    @staticmethod
    def generateClientNonce():
        '''
        Generate a client nonce.
        '''
        return TorControlProtocol.generateNonce(
            TorControlProtocol.SAFECOOKIE_CLIENT_NONCE_GENERATED_LENGTH)

    @staticmethod
    def generateServerNonce():
        '''
        Generate a server nonce.
        '''
        return TorControlProtocol.generateNonce(
            TorControlProtocol.SAFECOOKIE_SERVER_NONCE_LENGTH)

    # The ClientNonce, ServerHash, and ServerNonce values are
    # encoded/decoded in the same way as the argument passed to the
    # AUTHENTICATE command.

    @staticmethod
    def encodeNonce(nonce_bytes):
        '''
        Encode a nonce for transmission.
        '''
        encoded_nonce = TorControlProtocol.encodeControllerString(nonce_bytes)
        # sanity check
        assert (TorControlProtocol.decodeNonce(
                encoded_nonce, len(nonce_bytes), len(nonce_bytes)) ==
                nonce_bytes)
        return encoded_nonce

    # Use aliases for documentation purposes, and to match decoding functions
    encodeHash = encodeNonce
    encodeClientNonce = encodeNonce
    encodeServerNonce = encodeNonce
    encodeClientHash = encodeHash
    encodeServerHash = encodeHash

    @staticmethod
    def decodeNonce(encoded_str, min_len, max_len):
        '''
        Decode and check a received nonce.
        Returns the nonce if valid, or None if not valid.
        '''
        assert min_len >= 0
        assert max_len >= min_len
        decoded_bytes = TorControlProtocol.decodeControllerString(encoded_str)
        if len(decoded_bytes) < min_len:
            logging.warning("Received nonce was {} bytes, wanted at least {} bytes"
                            .format(len(decoded_bytes), min_len))
            return None
        if len(decoded_bytes) > max_len:
            logging.warning("Received nonce was {} bytes, wanted no more than {} bytes"
                            .format(len(decoded_bytes), max_len))
            return None
        return decoded_bytes

    # Use aliases for documentation purposes, and to match decoding functions
    decodeHash = decodeNonce

    @staticmethod
    def decodeClientNonce(encoded_str):
        '''
        Decode and check a received client nonce.
        Returns the nonce if valid, or None if not valid.
        '''
        return TorControlProtocol.decodeNonce(encoded_str,
                   TorControlProtocol.SAFECOOKIE_CLIENT_NONCE_MIN_VALID_LENGTH,
                   TorControlProtocol.SAFECOOKIE_CLIENT_NONCE_MAX_VALID_LENGTH)

    @staticmethod
    def decodeServerNonce(encoded_str):
        '''
        Decode and check a received server nonce.
        Returns the nonce if valid, or None if not valid.
        '''
        # ServerNonce MUST be 32 bytes long.
        return TorControlProtocol.decodeNonce(encoded_str,
                           TorControlProtocol.SAFECOOKIE_SERVER_NONCE_LENGTH,
                           TorControlProtocol.SAFECOOKIE_SERVER_NONCE_LENGTH)

    @staticmethod
    def decodeClientHash(encoded_str):
        '''
        Decode and check a received client hash.
        Returns the hash if valid, or None if not valid.
        '''
        # ClientHash MUST be 32 bytes long.
        return TorControlProtocol.decodeHash(encoded_str,
                              TorControlProtocol.SAFECOOKIE_CLIENT_HASH_LENGTH,
                              TorControlProtocol.SAFECOOKIE_CLIENT_HASH_LENGTH)

    @staticmethod
    def decodeServerHash(encoded_str):
        '''
        Decode and check a received server hash.
        Returns the hash if valid, or None if not valid.
        '''
        # ServerHash MUST be 32 bytes long.
        return TorControlProtocol.decodeHash(encoded_str,
                              TorControlProtocol.SAFECOOKIE_SERVER_HASH_LENGTH,
                              TorControlProtocol.SAFECOOKIE_SERVER_HASH_LENGTH)

    @staticmethod
    def getServerHash(cookie_string, client_nonce, server_nonce):
        '''
        Returns a SAFECOOKIE server hash using cookie_string, client_nonce,
        and server_nonce.
        '''
        # ServerHash is computed as:
        # HMAC-SHA256(
        #   "Tor safe cookie authentication server-to-controller hash",
        #   CookieString | ClientNonce | ServerNonce)
        server_hash = get_hmac(
            TorControlProtocol.SAFECOOKIE_SERVER_HASH_KEY,
            cookie_string, client_nonce + server_nonce)
        assert TorControlProtocol.verifyServerHash(server_hash, cookie_string,
                                                   client_nonce, server_nonce)
        return server_hash

    @staticmethod
    def verifyServerHash(server_hash, cookie_string, client_nonce,
                         server_nonce):
        '''
        Verifies a SAFECOOKIE server_hash using cookie_string, client_nonce,
        and server_nonce.
        Returns True if valid, False if invalid.
        '''
        if (server_hash is None or cookie_string is None or
            client_nonce is None or server_nonce is None):
            return False
        # Check using a timing-safe function
        # (the rest of the code is likely not timing-safe)
        return verify_hmac(server_hash,
                           TorControlProtocol.SAFECOOKIE_SERVER_HASH_KEY,
                           cookie_string, client_nonce + server_nonce)

    @staticmethod
    def getClientHash(cookie_string, client_nonce, server_nonce):
        '''
        Returns a SAFECOOKIE client hash using cookie_string, client_nonce,
        and server_nonce.
        '''
        # ClientHash is computed as:
        # HMAC-SHA256(
        #   "Tor safe cookie authentication controller-to-server hash",
        #   CookieString | ClientNonce | ServerNonce)
        client_hash = get_hmac(
            TorControlProtocol.SAFECOOKIE_CLIENT_HASH_KEY,
            cookie_string, client_nonce + server_nonce)
        assert TorControlProtocol.verifyClientHash(client_hash, cookie_string,
                                                   client_nonce, server_nonce)
        return client_hash

    @staticmethod
    def verifyClientHash(client_hash, cookie_string, client_nonce,
                         server_nonce):
        '''
        Verifies a SAFECOOKIE client_hash using cookie_string, client_nonce,
        and server_nonce.
        Returns True if valid, False if invalid.
        '''
        if (client_hash is None or cookie_string is None or
            client_nonce is None or server_nonce is None):
            return False
        # Check using a timing-safe function
        # (the rest of the code is likely not timing-safe)
        return verify_hmac(client_hash,
                           TorControlProtocol.SAFECOOKIE_CLIENT_HASH_KEY,
                           cookie_string, client_nonce + server_nonce)

    def setDiscoveredValue(self, set_function_name, value, value_name):
        '''
        When we discover a value, call factory.set_function_name to set it,
        and log a message containing value_name if this fails.
        '''
        try:
            # Equivalent to self.factory.set_function_name(value)
            if not getattr(self.factory, set_function_name)(value):
                logging.warning("Connection with {}: bad {} set via {}: {}"
                                .format(transport_info(self.transport),
                                        value_name, set_function_name, value))
        except AttributeError as e:
            logging.warning("Connection with {}: tried to set {} via {}: {}, but factory raised {}"
                            .format(transport_info(self.transport),
                                    value_name, set_function_name, value, e))

    def getConfiguredValue(self, get_function_name, value_name,
                           default=None):
        '''
        When we need a value, call factory.get_function_name to get it.
        Log a message containing value_name if this fails, and return default.
        '''
        try:
            # Equivalent to self.factory.get_function_name()
            return getattr(self.factory, get_function_name)()
        except AttributeError as e:
            logging.warning("Connection with {}: tried to get {} via {}, but factory raised {}, returning {}"
                            .format(transport_info(self.transport),
                                    value_name, get_function_name,
                                    e, default))
            return default

    # works for both configured and discovered values
    getDiscoveredValue = getConfiguredValue

    @staticmethod
    def readFile(secret_file, min_len, max_len):
        '''
        Read a value between min_len and max_len from secret_file.
        Return the value read from the file, or None if there is no file, or
        reading from the file fails, or the read secret is not an acceptable
        length.
        '''
        if secret_file is not None:
            try:
                with open(secret_file, 'r') as f:
                    # Read one more byte to check that the file is actually
                    # the right length
                    secret_string = f.read(max_len + 1)
            except IOError as e:
                logging.warning("Authentication failed: reading file '{}' failed with error: {}"
                                .format(secret_file, e))
                return None
            if len(secret_string) < min_len:
                logging.warning("Authentication failed, file '{}' was wrong length {}, wanted at least {}"
                                .format(secret_file,
                                        len(secret_string),
                                        min_len))
            if len(secret_string) > max_len:
                logging.warning("Authentication failed, file '{}' was wrong length {}, wanted at most {}"
                                .format(secret_file,
                                        len(secret_string),
                                        max_len))
                return None
            return secret_string
        else:
            return None

    def getConfiguredCookieFile(self):
        '''
        Return the configured path to the cookie file.
        Configuring more than one cookie file is not supported.
        '''
        return self.getConfiguredValue('get_control_cookie_file',
                                       'AuthCookieFile')

    def writeConfiguredCookieFile(self, cookie_string = None):
        '''
        Write a random 32-byte value to the configured cookie file.
        If cookie_string is not None, use that value.
        Return the value written to the file, or None if there is no cookie
        file, or if writing the file fails.
        '''
        cookie_file = self.getConfiguredCookieFile()
        if cookie_file is not None:
            if cookie_string is None:
                cookie_string = urandom(TorControlProtocol.SAFECOOKIE_LENGTH)
            try:
                with open(cookie_file, 'w') as f:
                    f.write(cookie_string)
            except IOError as e:
                logging.warning("Disabling SAFECOOKIE authentication, writing cookie file '{}' failed with error: {}"
                                .format(cookie_file, e))
                return None
            # sanity check: this will fail in write-only environments
            assert cookie_string == TorControlProtocol.readCookieFile(
                cookie_file)
            return cookie_string
        else:
            return None

    @staticmethod
    def readCookieFile(cookie_file):
        '''
        Read a 32-byte value from cookie_file.
        Return the value read from the file, or None if there is no cookie
        file, or reading from the file fails, or the cookie is not 32 bytes.
        '''
        # All authentication cookies are 32 bytes long.  Controllers
        # MUST NOT use the contents of a non-32-byte-long file as an
        # authentication cookie.
        return TorControlProtocol.readFile(
            cookie_file,
            TorControlProtocol.SAFECOOKIE_LENGTH,
            TorControlProtocol.SAFECOOKIE_LENGTH)

    def getConfiguredPasswordFile(self):
        '''
        Return the configured path to the password file.
        Configuring more than one password file is not supported.
        '''
        return self.getConfiguredValue('get_control_password',
                                       'password file')

    def getConfiguredPassword(self):
        '''
        Read a string from the configured password file.
        Return the value read from the file, or None if there is no password
        file, or reading from the file fails, or the password is too short
        (or too long).
        '''
        # PrivCount expects passwords between 8 bytes and 1 kilobyte
        # The hash used by tor outputs 20 bytes
        password_file = self.getConfiguredPasswordFile()
        return TorControlProtocol.readFile(
            password_file,
            TorControlProtocol.PASSWORD_MIN_VALID_LENGTH,
            TorControlProtocol.PASSWORD_MAX_VALID_LENGTH)

    def quit(self):
        '''
        Quit and close the connection
        Overridden in subclasses
        '''
        self.clear()

    def get_warn_length(self, is_line_received):
        '''
        Returns the line length at which we should warn, based on whether
        is_line_received or not
        '''
        # We put a strict limit on this, because it's harder to change
        # (and it uses integers, addresses, and DNS names, which are
        # often ~1/10 of their maximum length)
        return self.MAX_LENGTH / 10

    def check_line_length(self, line, is_line_received, is_length_exceeded):
        '''
        Warns on over-length lines, based on whether the line is received,
        and whether the line was delivered via lineLengthExceeded or not
        (sometimes, twisted's lineLengthExceeded only delivers a partial line,
        https://twistedmatrix.com/trac/ticket/6558
        and it had issues counting end of line characters
        https://twistedmatrix.com/trac/ticket/6536
        )
        Terminates the reactor if the line is over-length.
        '''
        is_length_exceeded = is_length_exceeded or len(line) > self.MAX_LENGTH
        is_unsafe_length = is_length_exceeded or len(line) > self.get_warn_length(is_line_received)
        # if we are over the safe length, warn
        if is_unsafe_length:
            logging.warning("{} line of length {} exceeded {} of {}, {} connection to {}"
                            .format("Received" if is_line_received else "Generated",
                                    len(line),
                                    "MAX_LENGTH" if is_length_exceeded else "safe length",
                                    self.get_warn_length(is_line_received),
                                    "dropping" if is_length_exceeded and is_line_received else "keeping",
                                    transport_info(self.transport)))
        # if we send or receive an overlength line, fail
        if is_length_exceeded:
            stop_reactor(1)

class TorControlClientProtocol(LineOnlyReceiver, TorControlProtocol):

    def __init__(self, factory):
        # we want ancestors to be able to use this in clear()
        self.consensus_refresher = None
        TorControlProtocol.__init__(self, factory)
        # we only want to clear this at the end of a round
        self.collection_events = None
        self.clear()

    def clear(self):
        '''
        Clear all the instance variables
        '''
        # this does a double-clear on init, that's ok
        TorControlProtocol.clear(self)
        self.state = None
        self.auth_methods = None
        self.cookie_file = None
        self.active_events = None
        self.has_received_events = False
        # Stop updating the flags when we're disconnected
        if self.consensus_refresher is not None:
            self.consensus_refresher.stop()
            self.consensus_refresher = None
        self.is_processing_ns = False

    def connectionMade(self):
        '''
        Ask for the available authentication methods over the control port,
        and put the protocol in the 'protocolinfo' state.
        Overrides twisted function.
        '''
        logging.info("Connection with {} was made"
                     .format(transport_info(self.transport)))
        self.sendLine("PROTOCOLINFO 1")
        self.state = 'protocolinfo'

    def isConnected(self):
        '''
        Is this procotol connected?
        '''
        return self.state is not None and self.state != 'disconnected'

    def startCollection(self, counter_list, event_list=None):
        '''
        Enable events for all the events required by counter_list, and all
        events explictly specified in event_list. After every successful
        control port connection, re-enable the events.
        '''
        if self.has_received_events:
            logging.warning("startCollection called multiple times without stopCollection")
            self.has_received_events = False
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
        logging.info("Starting PrivCount collection with {} events: {} from {} counters: {} and {} events: {}"
                     .format(len(self.collection_events),
                             " ".join(self.collection_events),
                             0 if counter_list is None else len(counter_list),
                             "(none)" if counter_list is None else
                             summarise_string(" ".join(counter_list), 100),
                             0 if event_list is None else len(event_list),
                             "(none)" if event_list is None else
                             " ".join(event_list)))
        logging.debug("Starting PrivCount collection with {} events: {} from {} counters (full list): {} and {} events: {}"
                      .format(len(self.collection_events),
                              " ".join(self.collection_events),
                              0 if counter_list is None else len(counter_list),
                              "(none)" if counter_list is None else
                              " ".join(counter_list),
                              0 if event_list is None else len(event_list),
                              "(none)" if event_list is None else
                              " ".join(event_list)))
        self.enableEvents()

    def stopCollection(self, clear_events=True):
        '''
        Disable all events. Remain connected to the control port, but wait for
        the next collection to start. If clear_events is True, forget the last
        set of events we used.
        '''
        logging.info("Stopping collection, {} events received."
                     .format("some" if self.has_received_events else "no"))
        self.has_received_events = False
        if clear_events:
            self.collection_events = None
        self.disableEvents()
        # let the user know that we're waiting
        logging.info("Waiting for next PrivCount collection to start")

    def enableEvents(self):
        '''
        If we are ready to handle events, and we know which events we want
        to process, turn privcount internals on, and turn on the events for
        the stored list of counters.
        '''
        # only start PrivCount if we are ready to handle events,
        # and know which events we want to handle
        if self.collection_events is None:
            logging.warning('Not enabling events: no events selected')
            return
        if not self.isConnected():
            logging.info('Not enabling events: not connected yet')
            return
        if self.state == 'waiting' or self.state == 'processing':
            self.active_events = self.collection_events
            use_setconf = self.getConfiguredValue('get_use_setconf',
                                                  'use SETCONF',
                                                  default=True)
            if use_setconf:
                circuit_sample_rate = self.getConfiguredValue(
                                                  'get_circuit_sample_rate',
                                                  'circuit sample rate',
                                                  default=1.0)
                max_cell_events_per_circuit = self.getConfiguredValue(
                                                  'get_max_cell_events_per_circuit',
                                                  'max cell events per circuit',
                                                  default=-1)
                # Protect the EnablePrivCount setting from logrotate and
                # similar
                self.sendLine("SETCONF __ReloadTorrcOnSIGHUP=0")
                # Avoid format string vulnerabilities
                circuit_sample_rate = float(circuit_sample_rate)
                max_cell_events_per_circuit = int(max_cell_events_per_circuit)
                # be tolerant of Tor versions without this feature
                if circuit_sample_rate != 1.0:
                    self.sendLine("SETCONF PrivCountCircuitSampleRate={}"
                                  .format(circuit_sample_rate))
                if max_cell_events_per_circuit >= 0:
                    self.sendLine("SETCONF PrivCountMaxCellEventsPerCircuit={}"
                                  .format(max_cell_events_per_circuit))
                self.sendLine("SETCONF EnablePrivCount=1")
            # Always check that EnablePrivCount is set, even if we just set it
            self.sendLine("GETCONF EnablePrivCount")
            # SETEVENTS is fine, it only affects this control connection
            self.sendLine("SETEVENTS {}".format(" ".join(self.active_events)))
            self.state = 'processing'
            logging.info("Enabled PrivCount events: {}"
                         .format(" ".join(self.active_events)))
        else:
            logging.warning('Not enabling events: in state {}'
                            .format(self.state))

    def disableEvents(self):
        '''
        Turn privcount events and privcount internals off
        '''
        # try to turn events off regardless of the state, as long as we have
        # connected and authenticated, this will work
        # (and if it doesn't, we haven't lost anything)
        if not self.isConnected():
            logging.info('Not disabling events, not connected')
        else:
            if self.active_events is not None:
                logging.info("Disabled PrivCount events: {}"
                             .format(" ".join(self.active_events)))
            self.sendLine("SETEVENTS")
            use_setconf = self.getConfiguredValue('get_use_setconf',
                                                  'use SETCONF',
                                                  default=True)
            if use_setconf:
                circuit_sample_rate = self.getConfiguredValue(
                                                  'get_circuit_sample_rate',
                                                  'circuit sample rate',
                                                  default=1.0)
                max_cell_events_per_circuit = self.getConfiguredValue(
                                                  'get_max_cell_events_per_circuit',
                                                  'max cell events per circuit',
                                                  default=-1)
                self.sendLine("SETCONF EnablePrivCount=0")
                # be tolerant of Tor versions without this feature
                if circuit_sample_rate != 1.0:
                    self.sendLine("SETCONF PrivCountCircuitSampleRate=1.0")
                if max_cell_events_per_circuit >= 0:
                    self.sendLine("SETCONF PrivCountMaxCellEventsPerCircuit=-1")
                self.sendLine("SETCONF __ReloadTorrcOnSIGHUP=1")
            # Don't check if EnablePrivCount is off: other instances might
            # want it to stay on
            self.active_events = None
            self.state = 'waiting'

    def sendLine(self, line):
        '''
        overrides twisted function
        '''
        logging.debug("Sending line '{}' to {}"
                      .format(line, transport_info(self.transport)))
        self.check_line_length(line, False, False)
        # make sure we don't issue a SETCONF when we're not supposed to
        if line.startswith("SETCONF"):
            use_setconf = self.getConfiguredValue('get_use_setconf',
                                                  'use SETCONF',
                                                  default=True)
            if not use_setconf:
                logging.warning("Connection with {}: protocol tried to use SETCONF when use_setconf was False: '{}'"
                            .format(transport_info(self.transport), line))
                self.quit()
                return
        return LineOnlyReceiver.sendLine(self, line)

    def sendInfoNs(self):
        '''
        Send a GETINFO ns/id/fingerprint request, where fingerprint is our
        discovered fingerprint.
        If there is no discovered fingerprint, quit.
        '''
        if self.state != 'waiting' and self.state != 'processing':
            logging.warning("Ignoring request to send GETINFO ns in state {}"
                            .format(self.state))
            return
        # get the validated fingerprint
        fingerprint = self.getDiscoveredValue('get_fingerprint',
                                              'fingerprint')
        if fingerprint is not None:
            self.sendLine("GETINFO ns/id/{}".format(fingerprint))
        else:
            self.quit()

    def lineReceived(self, line):
        '''
        Check that protocolinfo was successful.
        If so, authenticate using the best authentication method.
        Then, put the protocol in the 'discovering' state, and ask the relay
        for information about itself.
        When the fingerprint is received, put the protocol in the 'waiting'
        state, and ask the relay for its own consensus entry.
        When the consensus entry is received in the 'waiting' or 'processing'
        states, temporarily set is_processing_ns, and clear it when done.
        When the round is started, put the protocol in the 'processing' state,
        and send the list of events we want.
        When events are received, process them.
        Overrides twisted function.
        '''
        logging.debug("Received line '{}' from {}"
                      .format(line, transport_info(self.transport)))
        self.check_line_length(line, True, False)
        line = line.strip()

        if self.state == 'protocolinfo':
            if line.startswith("250-PROTOCOLINFO"):
                # 250-PROTOCOLINFO 1
                # ignore the protocolinfo version
                pass
            elif line.startswith("250-AUTH"):
                # 250-AUTH METHODS=AuthMethod,AuthMethod,... COOKIEFILE="AuthCookieFile"
                _, _, suffix = line.partition("METHODS=")
                methods, sep, cookie_file = suffix.partition("COOKIEFILE=")
                # if there is no cookie file
                if len(sep) == 0:
                    methods = suffix
                # save the supported methods for later
                self.auth_methods = methods.strip().split(",")
                # warn the user about security
                if "NULL" in self.auth_methods:
                    logging.warning("Your Tor control port has no authentication. Please configure CookieAuthentication or HashedControlPassword.")
                # if there is a cookie file that is not a quoted empty string
                if len(cookie_file) > 2:
                    # save the cookie file for later, stripping off trailing
                    # spaces and quotes (in that order, and only that order)
                    self.cookie_file = \
                        TorControlProtocol.decodeControllerString(cookie_file)
            elif line.startswith("250-VERSION"):
                # This version does *not* have the git tag
                # 250-VERSION Tor="TorVersion" OptArguments
                _, _, suffix = line.partition("Tor=")
                version, _, _ = suffix.partition(" ")
                # if there is a version that is not a quoted empty string
                if len(version) > 2:
                    version = \
                        TorControlProtocol.decodeControllerString(version)
                    # tell the factory
                    self.setDiscoveredValue('set_tor_version', version,
                                            'PROTOCOLINFO version')
            elif line == "250 OK":
                # we must authenticate as soon as we can
                password = self.getConfiguredPassword()
                if ("SAFECOOKIE" in self.auth_methods and
                    self.cookie_file is not None):
                    # send AUTHCHALLENGE, then AUTHENTICATE in response to
                    # AUTHCHALLENGE
                    self.client_nonce = \
                        TorControlProtocol.generateClientNonce()
                    encoded_client_nonce = \
                        TorControlProtocol.encodeClientNonce(self.client_nonce)
                    self.sendLine("AUTHCHALLENGE SAFECOOKIE {}"
                                  .format(encoded_client_nonce))
                    self.state = "authchallenge"
                elif ("HASHEDPASSWORD" in self.auth_methods and
                    password is not None):
                    encoded_password = \
                        TorControlProtocol.encodeControllerString(password)
                    self.sendLine("AUTHENTICATE " + encoded_password)
                    self.state = "authenticating"
                elif "NULL" in self.auth_methods:
                    # Authenticate without a password or cookie
                    logging.info("Authenticating with {} using {} method"
                                 .format(transport_info(self.transport),
                                         "NULL"))
                    self.sendLine("AUTHENTICATE")
                    self.state = "authenticating"
                else:
                    raise NotImplementedError("Authentication methods {} not implemented"
                                              .format(",".join(self.auth_methods)))
            else:
                # log any other lines at the appropriate level
                self.handleUnexpectedLine(line)
        elif (self.state == 'authchallenge' and
              line.startswith("250 AUTHCHALLENGE SERVERHASH=")):
            _, _, suffix = line.partition("250 AUTHCHALLENGE SERVERHASH=")
            server_hash, _, server_nonce = suffix.partition(" SERVERNONCE=")
            self.server_nonce = TorControlProtocol.decodeServerNonce(
                server_nonce)
            server_hash = TorControlProtocol.decodeServerHash(
                server_hash)
            # if the nonce or hash are invalid, we don't want to read the file
            if self.server_nonce is None or server_hash is None:
                logging.warning("Connection with {}: invalid AUTHCHALLENGE line: '{}'"
                            .format(transport_info(self.transport), line))
                self.quit()
                return
            self.cookie_string = TorControlProtocol.readCookieFile(
                self.cookie_file)
            server_hash_matches = TorControlProtocol.verifyServerHash(
                server_hash, self.cookie_string,
                self.client_nonce, self.server_nonce)
            if server_hash_matches:
                # Now we can authenticate
                client_hash = TorControlProtocol.getClientHash(
                    self.cookie_string, self.client_nonce, self.server_nonce)
                encoded_client_hash = \
                    TorControlProtocol.encodeClientHash(client_hash)
                self.sendLine("AUTHENTICATE " + encoded_client_hash)
                self.state = "authenticating"
            else:
                logging.warning("Connection with {}: bad AUTHCHALLENGE server hash or cookie file: '{}'"
                            .format(transport_info(self.transport), line))
                self.quit()
                return
        elif self.state == 'authenticating' and line == "250 OK":
            # Turn off privcount events, so we don't get spurious responses
            # We can't do this until we've authenticated (and we shouldn't
            # need to - there are no events sent before authentication)
            self.disableEvents()
            # Just ask for all the info all at once
            self.sendLine("GETCONF Nickname")
            self.sendLine("GETCONF ORPort")
            self.sendLine("GETCONF DirPort")
            self.sendLine("GETINFO version")
            self.sendLine("GETINFO address")
            # Unpatched Tor will break the protocol in response to this:
            self.sendLine("GETINFO privcount-version")
            self.sendLine("GETINFO fingerprint")
            self.state = 'discovering'
        elif self.state == 'discovering':
            # -- These cases continue discovering, or quit() -- #
            # It doesn't have PrivCount
            if line.startswith("552 Unrecognized option"):
                logging.critical("Connection with {}: does not support PrivCount"
                                 .format(transport_info(self.transport)))
                self.quit()
                return
            # Check PrivCount is on
            elif line.startswith("250 EnablePrivCount="):
                _, _, privcount_enabled = line.partition("EnablePrivCount=")
                if privcount_enabled != "1":
                    logging.warning("Connection with {} failed: EnablePrivCount is {}"
                                .format(transport_info(self.transport),
                                        privcount_enabled))
                    self.quit()
                    return
                else:
                    logging.info("Confirmed that EnablePrivCount is 1")
            # It's a relay, and it's just told us its Nickname
            elif line.startswith("250 Nickname="):
                _, _, nickname = line.partition("Nickname=")
                self.setDiscoveredValue('set_nickname', nickname, 'Nickname')
            # It doesn't have a Nickname, maybe it's a client?
            # But we'll catch that when we check the fingerprint, so just ignore this response
            elif line == "250 Nickname":
                logging.info("Connection with {}: no Nickname"
                             .format(transport_info(self.transport)))
                # It has no nickname, which is different to not knowing if it
                # has a nickname or not
                self.setDiscoveredValue('set_nickname', '', 'Nickname')
            # It's a relay, and it's just told us one of its ORPorts
            elif (line.startswith("250 ORPort=") or
                  line.startswith("250-ORPort=")):
                _, _, orport = line.partition("ORPort=")
                self.setDiscoveredValue('set_orport', orport, 'ORPort')
            # It doesn't have an ORPort, maybe it's a client?
            # But we'll catch that when we check the fingerprint, so just ignore this response
            elif line == "250 ORPort":
                logging.warning("Connection with {}: no ORPort"
                                .format(transport_info(self.transport)))
            # It's a relay, and it's just told us one of its DirPorts
            elif (line.startswith("250 DirPort=") or
                  line.startswith("250-DirPort=")):
                _, _, dirport = line.partition("DirPort=")
                self.setDiscoveredValue('set_dirport', dirport, 'DirPort')
            elif line == "250 DirPort":
                logging.info("Connection with {}: no DirPort"
                             .format(transport_info(self.transport)))
            # It's just told us its version
            # The control spec assumes that Tor always has a version, so there's no error case
            # This version *might* have a git tag, unlike the version
            # provided by PROTOCOLINFO. We want the git tag if possible,
            # because it uniquely identifies a privcount-patched tor version
            # (the data collector and tally server accept version changes)
            elif line.startswith("250-version="):
                _, _, version = line.partition("version=")
                self.setDiscoveredValue('set_tor_version', version,
                                        'GETINFO version')
            # It's just told us its address
            elif line.startswith("250-address="):
                _, _, address = line.partition("address=")
                self.setDiscoveredValue('set_address', address, 'address')
            # We asked for its address, and it couldn't find it. That's weird.
            elif line == "551 Address unknown":
                logging.info("Connection with {}: does not know its own address"
                             .format(transport_info(self.transport)))
            # It's a PrivCount-patched Tor instance, and it's just told us its
            # PrivCount version
            elif line.startswith("250-privcount-version="):
                _, _, version = line.partition("privcount-version=")
                self.setDiscoveredValue('set_tor_privcount_version', version,
                                        'GETINFO privcount-version')
            # We asked for its PrivCount version, and it didn't know what we
            # meant
            elif line == '552 Unrecognized key "privcount-version"':
                logging.warning("Connection with {} failed: not a PrivCount relay"
                                .format(transport_info(self.transport)))
                self.quit()
                return
            # -- fingerprint discovery ends in consensus or quit() --
            # It's a relay, and it's just told us its fingerprint
            elif line.startswith("250-fingerprint="):
                _, _, fingerprint = line.partition("fingerprint=")
                self.setDiscoveredValue('set_fingerprint', fingerprint,
                                        'fingerprint')
                # set the state in case the looping call is immediate
                self.state = 'waiting'

                # refresh our relay flags every consensus interval
                wait_time = 60*60
                self.consensus_refresher = task.LoopingCall(self.sendInfoNs)
                consensus_deferred = self.consensus_refresher.start(wait_time,
                                                                    now=True)
                consensus_deferred.addErrback(errorCallback)

                # If we already know what events we should have enabled, start
                # processing them
                self.enableEvents()
            # We asked for its fingerprint, and it said it's a client
            elif line == "551 Not running in server mode":
                logging.warning("Connection with {} failed: not a relay"
                                .format(transport_info(self.transport)))
                self.quit()
                return
            else:
                self.handleUnexpectedLine(line)
        # consensus processing: waiting or processing states
        # handle other events interspersed between these lines
        elif line.startswith("250-ns/id/") or line.startswith('552 Unrecognized key "ns/id/'):
            _, _, fingerprint = line.partition('ns/id/')
            fingerprint, _, _ = fingerprint.partition('=')
            fingerprint, _, _ = fingerprint.partition('"')
            logging.warning("Missing entry for relay {} in consensus, will check again in an hour"
                            .format(fingerprint))
            # clear the flag list
            self.setDiscoveredValue('set_flag_list',
                                    "",
                                    'our consensus flags')
        elif line.startswith("250+ns/id/"):
            # The intro line for our consensus entry
            self.is_processing_ns = True
        elif self.is_processing_ns and line.startswith("s "):
            # The flags line for our consensus entry
            _, _, flag_string = line.partition(" ")
            self.setDiscoveredValue('set_flag_list',
                                    flag_string,
                                    'our consensus flags')
        elif self.is_processing_ns and line.startswith("."):
            # The end line for our consensus entry
            self.is_processing_ns = False
        elif self.state == 'waiting' and line.startswith("2"):
            # log ok events while we're waiting for round start
            self.handleUnexpectedLine(line)
        elif self.state == 'processing' and line.startswith("650 PRIVCOUNT_"):
            parts = line.split(" ")
            assert len(parts) > 1
            # log the event
            self.has_received_events = True
            logging.debug("receiving event '{}'".format(line))
            # skip unwanted events
            if not parts[1] in self.active_events:
                if not parts[1] in get_valid_events():
                    logging.warning("Unknown event type {}".format(parts[1]))
                else:
                    logging.warning("Unwanted event type {}".format(parts[1]))
            # skip empty events
            elif len(parts) <= 2:
                logging.warning("Event with no data {}".format(line))
            # send the event, including the event type
            elif not self.factory.handle_event(parts[1:]):
                logging.warning("Rejected event {}".format(line))
                self.quit()
                return
        elif self.is_processing_ns:
            # ignore unexpected lines
            pass
        else:
            self.handleUnexpectedLine(line)

    def handleUnexpectedLine(self, line):
        '''
        Log any unexpected responses at an appropriate level.
        Quit on error responses.
        '''
        if line == "250 OK":
            logging.debug("Connection with {}: ok response: '{}'"
                          .format(transport_info(self.transport), line))
        elif line.startswith("650 PRIVCOUNT_"):
            logging.warning("Connection with {}: unexpected event: '{}'"
                            .format(transport_info(self.transport), line))
        elif line.startswith("5"):
            logging.warning("Connection with {}: unexpected response: '{}'"
                            .format(transport_info(self.transport), line))
            self.quit()
        elif line.startswith("2"):
            logging.info("Connection with {}: ok response: '{}'"
                         .format(transport_info(self.transport), line))
        else:
            logging.warning("Connection with {}: unexpected response: '{}'"
                            .format(transport_info(self.transport), line))
            self.quit()

    def lineLengthExceeded(self, line):
        '''
        overrides twisted function
        '''
        self.check_line_length(line, True, True)
        return LineOnlyReceiver.lineLengthExceeded(self, line)

    def clearConnection(self, how, error=None, destination=None):
        '''
        Clear the connection, regardless of how it failed.
        Uses how, error, and destination in a log message, if present.
        '''
        if how != 'quit' and self.isConnected():
            logging.warning("{} control connection with {}{} unexpectedly {}: {}."
                            .format('Open',
                                    transport_info(self.transport),
                                    (' ' + destination) if destination is not None else '',
                                    how,
                                    error if error is not None else '(no error)'))
            # reason.getErrorMessage() is pretty useless
            # Try to get something more informative, unless there was no error
            if (error is not None and
                not error.startswith("Connection was closed cleanly")):
                log_error()
        else:
            logging.info("{} control connection with {}{} {}: {}."
                         .format('Open' if self.isConnected() else 'Disconnected',
                                 transport_info(self.transport),
                                 (' ' + destination) if destination is not None else '',
                                 how,
                                 error if error is not None else '(no error)'))
        if self.isConnected():
            # keep events for a reconnect
            self.stopCollection(clear_events=False)
            self.sendLine("QUIT")
            # Avoid a spurious reentrant warning from connectionLost
            self.state = "disconnected"
            # TODO: try to reconnect on failure? See #326
            # We don't currently try to reconnect, because that hides bugs
            self.transport.loseConnection()
            # Don't rely on the remote side to end the connection
            # Protocols should not send any more data once the connection has been
            # terminated
        self.clear()
        self.state = "disconnected"

    def quit(self):
        '''
        Quit and close the connection
        '''
        self.clearConnection('quit')

    def connectionLost(self, reason):
        '''
        overrides twisted function
        '''
        self.clearConnection('lost',
                             reason.getErrorMessage())

    def clientConnectionFailed(self, connector, reason):
        '''
        overrides twisted function: only for clients
        '''
        connector.stopConnecting()
        self.clearConnection('failed (client)',
                             reason.getErrorMessage(),
                             connector.getDestination())

    def clientConnectionLost(self, connector, reason):
        '''
        overrides twisted function: only for clients
        '''
        connector.stopConnecting()
        self.clearConnection('lost (client)',
                             reason.getErrorMessage(),
                             connector.getDestination())

class TorControlServerProtocol(LineOnlyReceiver, TorControlProtocol):
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
    PROTOCOLINFO 1
    250-PROTOCOLINFO 1
    250-AUTH METHODS=SAFECOOKIE COOKIEFILE="/var/run/tor/control.authcookie"
    250-VERSION Tor="0.2.8.9"
    250 OK
    AUTHENTICATE
    250 OK
    SETCONF EnablePrivCount=1
    250 OK
    SETCONF PrivCountCircuitSampleRate=0.5
    250 OK
    SETCONF PrivCountMaxCellEventsPerCircuit=25
    250 OK
    GETINFO
    250 OK
    GETINFO fingerprint
    250-fingerprint=5E54527B88A544E1A6CBF412A1DE2B208E7BF9A2
    250 OK
    GETINFO ns/id/5E54527B88A544E1A6CBF412A1DE2B208E7BF9A2
    250+ns/id/3BE2E10936A852E7137E8A250CD324EBB246FA0C=
    r test000a O+LhCTaoUucTfoolDNMk67JG+gw BuWvIcuvDhapTNbrbHnglHluQ6c 2017-06-22 03:59:43 127.0.0.1 5000 7000
    s Authority Exit Fast Guard HSDir Running Stable V2Dir Valid
    w Bandwidth=0
    p reject 1-65535
    .
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
        TorControlProtocol.__init__(self, factory)
        self.clear()

    def clear(self):
        '''
        Clear all the instance variables
        '''
        # this does a double-clear on init, that's ok
        TorControlProtocol.clear(self)
        self.authenticated = False

    def connectionMade(self):
        '''
        overrides twisted function
        '''
        logging.debug("Connection with {} was made"
                      .format(transport_info(self.transport)))

    def sendLine(self, line):
        '''
        overrides twisted function
        '''
        logging.debug("Sending line '{}' to {}"
                      .format(line, transport_info(self.transport)))
        self.check_line_length(line, False, False)
        return LineOnlyReceiver.sendLine(self, line)

    def lineReceived(self, line):
        '''
        overrides twisted function
        '''
        logging.debug("Received line '{}' from {}"
                      .format(line, transport_info(self.transport)))
        self.check_line_length(line, True, False)
        line = line.strip()
        parts = line.split(' ')

        # Quit regardless of authentication state
        if parts[0] == "QUIT":
            self.quit()
            return

        # We use " quotes in some places where tor uses ' quotes.
        # This should not matter: where it is significant, we match tor's
        # use of " quotes.
        if not self.authenticated:
            config_password = self.getConfiguredPassword()
            # The controller is meant to do PROTOCOLINFO, AUTHCHALLENGE,
            # AUTHENTICATE. We don't enforce the order, or require that each
            # request is only made once.
            if len(parts) > 0 and parts[0] == "PROTOCOLINFO":
                # Assume the protocolinfo version is OK
                self.sendLine("250-PROTOCOLINFO 1")
                auth_methods = []
                # We don't do COOKIE authentication, it's not secure,
                # instead, we do SAFECOOKIE
                self.cookie_string = self.writeConfiguredCookieFile(
                    self.cookie_string)
                if self.cookie_string is not None:
                    auth_methods.append("SAFECOOKIE")
                if config_password is not None:
                    auth_methods.append("HASHEDPASSWORD")
                # We don't do NULL authentication unless there are no other
                # options (what would be the point, otherwise?)
                if len(auth_methods) == 0:
                    auth_methods.append("NULL")
                if self.cookie_string is not None:
                    encoded_cookie_file = \
                        TorControlProtocol.encodeControllerString(
                        self.getConfiguredCookieFile())
                    cookie_part = " COOKIEFILE={}".format(encoded_cookie_file)
                else:
                    cookie_part = ""
                self.sendLine("250-AUTH METHODS={}{}".format(",".join(auth_methods), cookie_part))
                self.sendLine("250-VERSION Tor=\"0.2.8.6\"")
                self.sendLine("250 OK")
            elif line.startswith("AUTHCHALLENGE SAFECOOKIE"):
                if self.cookie_string is None:
                    self.sendLine("513 SAFECOOKIE authentication is not enabled")
                    self.transport.loseConnection()
                    return
                else:
                    _, _, client_nonce = line.partition(
                        "AUTHCHALLENGE SAFECOOKIE")
                    decoded_client_nonce = None
                    try:
                        decoded_client_nonce = \
                            TorControlProtocol.decodeClientNonce(client_nonce)
                    except TypeError:
                        # it's an invalid nonce and will fail the next check
                        pass
                    if decoded_client_nonce is None:
                        self.sendLine("513 Invalid base16 client nonce")
                        self.transport.loseConnection()
                        return
                    self.client_nonce = decoded_client_nonce
                    self.server_nonce = \
                        TorControlProtocol.generateServerNonce()
                    server_hash = TorControlProtocol.getServerHash(
                        self.cookie_string, self.client_nonce,
                        self.server_nonce)
                    encoded_server_hash = TorControlProtocol.encodeServerHash(
                        server_hash)
                    encoded_server_nonce = \
                        TorControlProtocol.encodeServerNonce(self.server_nonce)
                    self.sendLine(
                        "250 AUTHCHALLENGE SERVERHASH={} SERVERNONCE={}"
                        .format(encoded_server_hash, encoded_server_nonce))
            elif line.startswith("AUTHCHALLENGE"):
                self.sendLine("513 AUTHCHALLENGE only supports SAFECOOKIE authentication")
                self.transport.loseConnection()
                return
            elif line.startswith("AUTHENTICATE"):
                _, _, client_password = line.partition("AUTHENTICATE")
                client_password = client_password.strip()
                try:
                    client_password = \
                        TorControlProtocol.decodeControllerString(
                        client_password)
                except TypeError:
                    self.sendLine("551 Invalid quoted string.  You need to put the password in double quotes.")
                    self.transport.loseConnection()
                    return
                client_hash_matches = False
                # Check the length before verifying: like decodeClientHash,
                # but doesn't warn on bad hashes
                if (len(client_password) ==
                    TorControlProtocol.SAFECOOKIE_CLIENT_HASH_LENGTH):
                    client_hash_matches = TorControlProtocol.verifyClientHash(
                        client_password, self.cookie_string,
                        self.client_nonce, self.server_nonce)
                # If there is no set password, accept any password or cookie
                # hash: this is safe, because this tor control protocol stub
                # does not handle any sensitive information
                if ((self.getConfiguredCookieFile() is None and
                     config_password is None) or
                    client_hash_matches or
                    config_password == client_password):
                    self.sendLine("250 OK")
                    self.authenticated = True
                else:
                    self.sendLine("515 Authentication failed: Password did not match HashedControlPassword *or* authentication cookie.")
                    self.transport.loseConnection()
                    return
            else:
                self.sendLine("514 Authentication required.")
                self.transport.loseConnection()
                return
        elif len(parts) > 0:
            if parts[0] == "SETEVENTS":
                # events are case-insensitive, so convert to uppercase
                upper_setevents = map(str.upper, parts[1:])
                # already uppercase
                upper_known_events = get_valid_events()
                upper_setevents = set(upper_setevents)
                if len(upper_setevents) == 0:
                    # if there are no requested events, turn events off
                    self.sendLine("250 OK")
                    self.factory.stop_injecting()
                elif upper_setevents.issubset(upper_known_events):
                    # if every requested event is in the known events
                    self.sendLine("250 OK")
                    self.factory.start_injecting()
                else:
                    unknown_events = upper_setevents.difference(
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
                elif len(parts) == 2 and parts[1] == "privcount-version":
                    self.sendLine("250-privcount-version={}"
                                  .format(PRIVCOUNT_SHORT_VERSION_STRING))
                    self.sendLine("250 OK")
                elif len(parts) == 2 and parts[1] == "address":
                    # TEST-NET from https://tools.ietf.org/html/rfc5737
                    self.sendLine("250-address=192.0.2.91")
                    self.sendLine("250 OK")
                elif len(parts) == 2 and parts[1] == "fingerprint":
                    self.sendLine("250-fingerprint=FACADE0000000000000000000123456789ABCDEF")
                    self.sendLine("250 OK")
                elif len(parts) == 2 and parts[1].startswith("ns/id/"):
                    # pretend we know about the relay, and send a multi-line
                    # response
                    self.sendLine("250+{}=".format(parts[1]))
                    # if you're doing fingerprint consistency checks, this
                    # will fail
                    self.sendLine("r test004r O+LhCTaoUucTfoolDNMk67JG+gw BuWvIcuvDhapTNbrbHnglHluQ6c 2017-06-22 03:59:43 127.0.0.1 5000 7000")
                    self.sendLine("s Exit Fast Guard HSDir Running Stable V2Dir Valid")
                    self.sendLine("w Bandwidth=100000")
                    self.sendLine("p accept 1-65535")
                    self.sendLine(".")
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
                    # yes, relays can have multiple ORPorts (and DirPorts)
                    self.sendLine("250-ORPort=9001")
                    self.sendLine("250 ORPort=[::1]:12345")
                elif len(parts) == 2 and parts[1].lower() == "dirport":
                    self.sendLine("250 DirPort=9030")
                elif len(parts) == 2 and parts[1].lower() == "enableprivcount":
                    self.sendLine("250 EnablePrivCount=1")
                else:
                    # Like GETINFO, our GETCONF does not accept multiple words
                    self.sendLine('552 Unrecognized configuration key "{}"'.format(parts[1]))
            elif parts[0] == "SETCONF":
                # the correct response to an empty SETCONF is an empty OK
                if len(parts) == 1:
                    self.sendLine("250 OK")
                # Like GETCONF, SETCONF is case-insensitive, and returns one
                # line: "250 OK"
                elif (len(parts) == 2 and
                      (parts[1].lower().startswith("enableprivcount") or
                       parts[1].lower().startswith("__reloadtorrconsighup") or
                       parts[1].lower().startswith("privcountmaxcelleventspercircuit") or
                       parts[1].lower().startswith("privcountcircuitsamplerate"))):
                    # just ignore the value
                    self.sendLine("250 OK")
                else:
                    # Like GETINFO, our GETCONF does not accept multiple words
                    # and it doesn't bother to strip the =
                    self.sendLine('552 Unrecognized option: Unknown option "{}". Failing.'.format(parts[1]))
            else:
                self.sendLine('510 Unrecognized command "{}"'.format(parts[0]))
        else:
            self.sendLine('510 Unrecognized command ""')

    def lineLengthExceeded(self, line):
        '''
        overrides twisted function
        '''
        self.check_line_length(line, True, True)
        return LineOnlyReceiver.lineLengthExceeded(self, line)

    def quit(self):
        '''
        Quit and close the connection
        '''
        self.factory.stop_injecting()
        self.sendLine("250 closing connection")
        self.transport.loseConnection()
        self.clear()

    def connectionLost(self, reason):
        '''
        overrides twisted function
        '''
        logging.debug("Connection with {} was lost: {}"
                      .format(transport_info(self.transport),
                              reason.getErrorMessage()))
        self.factory.stop_injecting()
        self.clear()
