'''
Created on Dec 15, 2015

@author: rob
'''
import sys
import struct
import traceback
import logging
import socket
import datetime
import uuid
import json

from subprocess import Popen, PIPE
from random import gauss, randint
from os import urandom, path
from math import sqrt, exp, log
from time import time, strftime, gmtime
from copy import deepcopy
from base64 import b64encode, b64decode

from hashlib import sha256 as Hash

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

def load_compressed_json(filename):
    cmd = "xz --decompress --stdout {0}".format(filename)
    xzproc = Popen(cmd.split(), stdout=PIPE)
    obj = json.load(xzproc.stdout)
    xzproc.wait()
    return obj

def load_private_key_string(key_string):
    return serialization.load_pem_private_key(key_string, password=None, backend=default_backend())

def load_private_key_file(key_file_path):
    with open(key_file_path, 'rb') as key_file:
        private_key = load_private_key_string(key_file.read())
    return private_key

def load_public_key_string(key_string):
    return serialization.load_pem_public_key(key_string, backend=default_backend())

def load_public_key_file(key_file_path):
    with open(key_file_path, 'rb') as key_file:
        public_key = load_public_key_string(key_file.read())
    return public_key

def get_public_bytes(key_string, is_private_key=True):
    if is_private_key:
        private_key = load_private_key_string(key_string)
        public_key = private_key.public_key()
    else:
        public_key = load_public_key_string(key_string)
    return public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)

def get_public_digest_string(key_string, is_private_key=True):
    return Hash(get_public_bytes(key_string, is_private_key)).hexdigest()

def get_public_digest(key_path, is_private_key=True):
    with open(key_path, 'rb') as key_file:
        digest = get_public_digest_string(key_file.read(), is_private_key)
    return digest

def get_serialized_public_key(key_path, is_private_key=True):
    with open(key_path, 'rb') as key_file:
        data = get_public_bytes(key_file.read(), is_private_key)
    return data

def encrypt(pub_key, plaintext):
    ciphertext = pub_key.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA1()),
            algorithm=hashes.SHA1(),
            label=None
        )
    )
    return b64encode(ciphertext)

def decrypt(priv_key, ciphertext):
    plaintext = priv_key.decrypt(
        b64decode(ciphertext),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA1()),
            algorithm=hashes.SHA1(),
            label=None
        )
    )
    return plaintext

def generate_keypair(key_out_path):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096, backend=default_backend())
    pem = private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption())
    with open(key_out_path, 'wb') as outf:
        print >>outf, pem

def generate_cert(key_path, cert_out_path):
    private_key = load_private_key_file(key_path)
    public_key = private_key.public_key()

    builder = x509.CertificateBuilder()
    builder = builder.subject_name(x509.Name([
        x509.NameAttribute(x509.OID_COMMON_NAME, u'PrivCount User'),
    ]))
    builder = builder.issuer_name(x509.Name([
        x509.NameAttribute(x509.OID_COMMON_NAME, u'PrivCount Authority'),
    ]))
    builder = builder.not_valid_before(datetime.datetime.today() - datetime.timedelta(days=1))
    builder = builder.not_valid_after(datetime.datetime(2020, 1, 1))
    builder = builder.serial_number(int(uuid.uuid4()))
    builder = builder.public_key(public_key)
    builder = builder.add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)

    certificate = builder.sign(private_key=private_key, algorithm=hashes.SHA256(), backend=default_backend())

    with open(cert_out_path, 'wb') as outf:
        print >>outf, certificate.public_bytes(encoding=serialization.Encoding.PEM)

def get_random_free_port():
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = randint(10000, 60000)
        rc = s.connect_ex(('127.0.0.1', port))
        s.close()
        if rc != 0: # error connecting, port is available
            return port

## File Paths ##

def normalise_path(path_str):
    '''
    Return the abolute path corresponding to path_str, with user directories
    expanded, and the current working directory assumed for relative paths
    '''
    expanded_path = path.expanduser(path_str)
    return path.abspath(expanded_path)

## Logging ##

def log_error():
    _, _, tb = sys.exc_info()
    #traceback.print_tb(tb) # Fixed format
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]
    logging.warning("An error occurred in file '%s', at line %d, in func %s, in statement '%s'", filename, line, func, text)

## Logging: Time Formatting Functions ##
## a timestamp is an absolute point in time, in seconds since unix epoch
## a period is a relative time duration, in seconds
## a time argument is either a period or a timestamp
## a desc argument is a string description of the timestamp's meaning
## All period and timestamp arguments are normalised using normalise_time()
## before any calculations or formatting are performed

def normalise_time(time):
    '''
    Return the normalised value of time
    An abstraction used for consistent time rounding behaviour
    '''
    # we ignore microseconds
    return int(time)

def current_time():
    '''
    Return the normalised value of the current time
    '''
    return normalise_time(time())

def format_period(period):
    '''
    Format a time period as a human-readable string
    period is in seconds
    Returns a string of the form:
    1w 3d 12h 20m 32s
    starting with the first non-zero period (seconds are always included)
    '''
    period = normalise_time(period)
    period_str = ""
    # handle negative times by prepending a minus sign
    if period < 0:
        period_str += "-"
        period = -period
    # there's no built-in way of formatting a time period like this in python.
    # strftime is almost there, but would have issues with year-long periods.
    # divmod gives us the desired floor division result, and the remainder,
    # which will be floating point if normalise_time() returns floating point
    (week,   period) = divmod(period, 7*24*60*60)
    (day,    period) = divmod(period,   24*60*60)
    (hour,   period) = divmod(period,      60*60)
    (minute, period) = divmod(period,         60)
    # if normalise_time yields floating point values (microseconds), this will
    # produce a floating point result, which will be formatted as NN.NN
    # if it's an integer, it will format as NN. This is the desired behaviour.
    second           =        period % (      60)
    # now build the formatted string starting with the first non-zero period
    larger_period = 0
    if week > 0:
        period_str += "{}w ".format(week)
        larger_period = 1
    if day > 0 or larger_period:
        period_str += "{}d ".format(day)
        larger_period = 1
    if hour > 0 or larger_period:
        period_str += "{}h ".format(hour)
        larger_period = 1
    if minute > 0 or larger_period:
        period_str += "{}m ".format(minute)
    # seconds are always included, even if they are zero, or if there is no
    # larger period
    period_str += "{}s".format(second)
    return period_str

def format_datetime(timestamp):
    '''
    Format a timestamp as a human-readable UTC date and time string
    timestamp is in seconds since the epoch
    Returns a string of the form:
    2016-07-16 17:58:00
    '''
    timestamp = normalise_time(timestamp)
    return strftime("%Y-%m-%d %H:%M:%S", gmtime(timestamp))

def format_epoch(timestamp):
    '''
    Format a timestamp as a unix epoch numeric string
    timestamp is in seconds since the epoch
    Returns a string of the form:
    1468691880
    '''
    timestamp = normalise_time(timestamp)
    return str(timestamp)

def format_time(period, desc, timestamp):
    '''
    Format a period and timestamp as a human-readable string in UTC
    period is in seconds, and timestamp is in seconds since the epoch
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-16 17:58:00 1468691880)
    '''
    return "{} ({} {} {})".format(format_period(period),
                                  desc,
                                  format_datetime(timestamp),
                                  format_epoch(timestamp))

def format_interval(period, desc, begin_timestamp, end_timestamp):
    '''
    Format a period and two interval timestamps as a human-readable string in UTC
    period is in seconds, and the timestamps are in seconds since the epoch
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-16 17:58:00 to 2016-07-27 06:18:32,
    1468691880 to 1469600312)
    '''
    return "{} ({} {} to {}, {} to {})".format(format_period(period),
                                               desc,
                                               format_datetime(begin_timestamp),
                                               format_datetime(end_timestamp),
                                               format_epoch(begin_timestamp),
                                               format_epoch(end_timestamp))

def format_elapsed_time_since(past_timestamp, desc):
    '''
    Format the time elapsed since a past event, and that event's time in UTC
    past_timestamp is in seconds since the epoch
    The elapsed time is from past_timestamp to the current time
    past_timestamp is typically status['time'], and desc is typically 'since'
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-16 17:58:00 1468691880)
    '''
    # Normalise before calculation to avoid truncation errors
    past_timestamp = normalise_time(past_timestamp)
    elapsed_period = current_time() - past_timestamp
    return format_time(elapsed_period, desc, past_timestamp)

def format_delay_time_wait(delay_period, desc):
    '''
    Format the time delay until a future event, and the expected event time
    in UTC
    delay_period is in seconds
    The event time is the current time plus delay_period
    delay_period is typically config['defer_time'], and desc is typically 'at'
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-27 06:18:32 1469600312)
    '''
    # Normalise before calculation to avoid truncation errors
    delay_period = normalise_time(delay_period)
    future_timestamp = current_time() + delay_period
    return format_time(delay_period, desc, future_timestamp)

def format_delay_time_until(future_timestamp, desc):
    '''
    Format the time delay until a future event, and the expected event time
    in UTC
    The time delay is the difference between future_timestamp and the current
    time
    future_timestamp is in seconds since the epoch
    future_timestamp is typically config['defer_time'], and desc is typically 'at'
    returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-27 06:18:32 1469600312)
    '''
    # Normalise before calculation to avoid truncation errors
    future_timestamp = normalise_time(future_timestamp)
    delay_period = future_timestamp - current_time()
    return format_time(delay_period, desc, future_timestamp)

def format_interval_time_between(begin_timestamp, desc, end_timestamp):
    '''
    Format the interval elapsed between two events, and the times of those
    events in UTC
    The timestamps are in seconds since the epoch
    The interval is between begin_time and end_time
    desc is typically 'from'
    Returns a string of the form:
    1w 3d 12h 20m 32s (desc 2016-07-16 17:58:00 to 2016-07-27 06:18:32,
    1468691880 to 1469600312)
    '''
    # Normalise before calculation to avoid truncation errors
    begin_timestamp = normalise_time(begin_timestamp)
    end_timestamp = normalise_time(end_timestamp)
    period = end_timestamp - begin_timestamp
    return format_interval(period, desc, begin_timestamp, end_timestamp)

def format_last_event_time_since(last_event_timestamp):
    '''
    Format the time elapsed since the last Tor event, and that event's time
    in UTC
    last_event_timestamp is in seconds since the epoch, and can be None
    for no events
    The elapsed time is from last_event_timestamp to the current time
    Returns a string in one of the following forms:
    no Tor events received
    last Tor event was 1w 3d 12h 20m 32s (at 2016-07-16 17:58:00 1468691880)
    '''
    if last_event_timestamp is None:
        return "no Tor events received"
    else:
        return "last Tor event was {}".format(format_elapsed_time_since(
                                                  last_event_timestamp, 'at'))

## Calculation ##

def counter_modulus():
    '''
    The hard-coded modulus value for a blinded counter
    Blinded counters are unsigned
    In PrivCount, this does not have to be prime, and there is no need for it
    to be configurable
    All PrivCount counters should use unlimited-length Python longs, so that
    counter_modulus can exceed 64 bits, the size of a native C long
    '''
    # historical q values
    #return 2147483647L
    #return 999999999959L
    # modulus is limited to 2**64, because sample() only unpacks 8 bytes
    return 2L**64L

def min_blinded_counter_value():
    '''
    The hard-coded minimum value for a blinded counter
    Blinded counters are unsigned
    Always zero
    '''
    return 0L

def max_blinded_counter_value():
    '''
    The hard-coded maximum value for a blinded counter
    Blinded counters are unsigned
    '''
    return counter_modulus() - 1L

def min_tally_counter_value():
    '''
    The hard-coded minimum value for a tallied counter
    Tallied counters are signed, to allow for negative noise
    '''
    return adjust_count_signed((counter_modulus() + 1)//2,
                               counter_modulus())

def max_tally_counter_value():
    '''
    The hard-coded maximum value for a tallied counter
    Tallied counters are signed, to allow for negative noise
    '''
    return adjust_count_signed((counter_modulus() + 1)//2 - 1,
                               counter_modulus())

def add_counter_limits_to_config(config):
    '''
    Add the hard-coded counter limits to a deep copy of the config dictionary
    Returns the modified deep copy of the config dictionary
    '''
    assert config is not None
    config = deepcopy(config)
    # call this modulus so it sorts near the other values
    config['modulus'] = counter_modulus()
    config['min_blinded_counter_value'] = min_blinded_counter_value()
    config['max_blinded_counter_value'] = max_blinded_counter_value()
    config['min_tally_counter_value'] = min_tally_counter_value()
    config['max_tally_counter_value'] = max_tally_counter_value()
    return config

def noise(sigma, sum_of_sq, p_exit):
    '''
    Sample noise from a gussian distribution
    the distribution is over +/- sigma, scaled by the noise weight
    calculated from the exit probability p_exit, and the overall sum_of_sq
    bandwidth
    returns a floating-point value between +sigma and -sigma, scaled by
    noise_weight
    '''
    sigma_i = p_exit * sigma / sqrt(sum_of_sq)
    random_sample = gauss(0, sigma_i)
    return random_sample

def PRF(key, IV):
    '''
    Calculate pseudo-random bytes using a keyed hash based on key and IV
    Given the same key and IV, the same pseudo-random bytes will be produced
    key must contain at least as many bits of entropy as the hash
    Therefore, it must be at least 32 bytes long
    returns 32 pseudo-random bytes
    '''
    assert len(key) >= Hash().digest_size
    prv = Hash("PRF1|KEY:%s|IV:%s|" % (key, IV)).digest()
    # for security, the key input must have at least as many bytes as the hash
    # output (we do not depend on the length or content of IV for security)
    assert len(key) >= len(prv)
    return prv

SAMPLE_BYTE_MAX = 8L
SAMPLE_BIT_MAX = SAMPLE_BYTE_MAX * 8L

def sample(s, modulus):
    '''
    Sample a uniformly pseudo-random value from the random byte array s,
    returning a value less than modulus
    s must be at least 8 bytes long
    the returned value has a maximum of 2**64 - 1, so modulus is limited to 2**64
    returns a uniformly distributed in [0, modulus)
    '''
    # in order for this function to return v in {0, ... , modulus-1}, modulus-1 must be
    # representable in SAMPLE_BIT_MAX bits or less
    # (2**N is the first number not representable in an N-bit unsigned integer)
    assert long(modulus-1) < 2L**SAMPLE_BIT_MAX
    ## Unbiased sampling through rejection sampling
    while True:
        # s must have enough bytes for us to extract 8
        assert len(s) >= SAMPLE_BYTE_MAX
        # to get a value of modulus-1, we need this many bits
        q_bit_count = long(modulus-1).bit_length()
        # and this many bytes
        q_byte_count = (q_bit_count+7)//8
        assert q_byte_count <= SAMPLE_BYTE_MAX
        # Pad the bytes we'll never use with zeroes, otherwise we reject
        # a large number of values
        # Since the extraction is little-endian, and we want to pad unused
        # bytes, the zero bytes go in the larger indexes
        s_bytes = s[:q_byte_count] + ('\0' * (SAMPLE_BYTE_MAX - q_byte_count))
        # <Q means unpack 8 little-endian bytes into a C unsigned long long
        v = long(struct.unpack("<Q", s_bytes)[0])
        # this will fail if we have mismatching endianness and padding
        assert v < 2L**(q_byte_count*8L)
        if 0L <= v < modulus:
            break
        # when we reject the value, re-hash s and try again
        s = Hash(s).digest()
    return v

def derive_blinding_factor(label, secret, modulus, positive=True):
    '''
    Calculate a blinding factor less than modulus, based on label and secret
    when positive is True, return the blinding factor, and when positive is
    False, returns the unblinding factor (the inverse value mod modulus)
    '''
    ## Keyed share derivation
    s = PRF(secret, label)
    v = sample(s, modulus)
    assert v < modulus
    s0 = v if positive else modulus - v
    return s0

def adjust_count_signed(count, modulus):
    '''
    adjust the unsigned 0 <= count < modulus, returning a signed integer
    for odd  modulus, returns { -modulus//2, ... , 0, ... , modulus//2 }
    for even modulus, returns { -modulus//2, ... , 0, ... , modulus//2 - 1 }
    with the smaller positive values >= modulus//2 [- 1] becoming the largest negative
    values
    this is the inverse operation of x % modulus, when x is in the appropriate range
    (x % modulus always returns a positive integer when modulus is positive)
    '''
    # sanity check input
    assert count < modulus
    # When implementing this adjustment,
    # { 0, ... , (modulus + 1)//2 - 1}  is interpreted as that value,
    # { (modulus + 1)//2, ... , modulus - 1 } is interpreted as that value minus modulus, or
    # { (modulus + 1)//2 - modulus, ... , modulus - 1 - modulus }
    #
    # For odd modulus, (modulus + 1)//2 rounds up to modulus//2 + 1, so positive simplifies to:
    # { 0, ... , modulus//2 + 1 - 1 }
    # { 0, ... , modulus//2 }
    # and because modulus == modulus//2 + modulus//2 + 1 for odd modulus, negative simplifies to:
    # { modulus//2 + 1 - modulus//2 - modulus//2 - 1, ... , modulus - 1 - modulus}
    # { -modulus//2, ... , -1 }
    # Odd modulus has the same number of values above and below 0:
    # { -modulus//2, ... , 0, ... , modulus//2 }
    #
    # For even modulus, (modulus+1)//2 rounds down to modulus//2, so positive simplifies to:
    # { 0, ... , modulus//2 - 1 }
    # and because modulus == modulus//2 + modulus//2 for even modulus, negative simplifies to:
    # { modulus//2 - modulus//2 - modulus//2, ... , modulus - 1 - modulus}
    # { -modulus//2, ... , -1 }
    # Even modulus has the 1 more value below 0 than above it:
    # { -modulus//2, ... , 0, ... , modulus//2 - 1 }
    # This is equivalent to signed two's complement, if modulus is an integral power
    # of two
    if count >= ((modulus + 1L) // 2L):
        signed_count = count - modulus
    else:
        signed_count = count
    # sanity check output
    assert signed_count >= -modulus//2L
    if modulus % 2L == 1L:
        # odd case
        assert signed_count <= modulus//2L
    else:
        # even case
        assert signed_count <= modulus//2L - 1L
    return signed_count

class SecureCounters(object):
    '''
    securely count any number of labels
    counters should be in the form like this:
    {
      'CircuitCellsInOutRatio': {
        'bins':
        [
          [0.0, 0.1],
          [0.1, 0.25],
          [0.25, 0.5],
          [0.5, 0.75],
          [0.75, 0.9],
          [0.9, 1.0],
          [1.0, float('inf')],
        ],
        'sigma': 2090007.68996
      },
      'CircuitCellsIn': {
        'bins':
        [
          [0.0, 512.0],
          [512.0, 1024.0],
          [1024.0, 2048.0],
          [2048.0, 4096.0],
          [4096.0, float('inf')],
        ],
        'sigma': 2090007.68996
      }
    }
    All of data collectors, share keepers, and tally server use this to store counters
    it is used approximately like this:

    data collector:
    init(), generate(), detach_blinding_shares(), increment()[repeated], detach_counts()
    the blinding shares are sent to each share keeper
    the counts are sent to the tally server at the end

    share keeper:
    init(), import_blinding_share()[repeated], detach_counts()
    import..() uses the shares from each data collector
    the counts are sent to the tally server at the end

    tally server:
    init(), tally_counters(), detach_counts()
    tally..() uses the counts received from all of the data collectors and share keepers
    this produces the final, unblinded, noisy counts of the privcount process

    see privcount/test/test_counters.py for some test cases
    '''

    def __init__(self, counters, modulus):
        self.counters = deepcopy(counters)
        self.modulus = long(modulus)
        self.shares = None

        # initialize all counters to 0L
        # counters use unlimited length integers to avoid overflow
        for key in self.counters:
            if 'bins' not in self.counters[key]:
                return None
            for item in self.counters[key]['bins']:
                assert len(item) == 2
                item.append(0L) # bin is now, e.g.: [0.0, 512.0, 0L] for bin_left, bin_right, count

    def _derive_all_counters(self, secret, positive):
        for key in self.counters:
            for item in self.counters[key]['bins']:
                label = "{}_{}_{}".format(key, item[0], item[1])
                blinding_factor = derive_blinding_factor(label, secret, self.modulus, positive=positive)
                item[2] = (item[2] + long(blinding_factor)) % self.modulus

    def _blind(self, secret):
        self._derive_all_counters(secret, True)

    def _unblind(self, secret):
        self._derive_all_counters(secret, False)

    def generate(self, uids, noise_weight):
        self.shares = {}
        for uid in uids:
            # the secret should be at least as large as the hash output
            secret = urandom(Hash().digest_size)
            hash_id = PRF(secret, "KEYID")
            self.shares[uid] = {'secret': secret, 'hash_id': hash_id}
            # add blinding factors to all of the counters
            self._blind(secret)

	      # Add noise for each counter independently
        for key in self.counters:
            for item in self.counters[key]['bins']:
                sigma = self.counters[key]['sigma']
                sampled_noise = noise(sigma, 1, noise_weight)
                noise_val = long(round(sampled_noise))
                # if noise_val is negative, modulus produces a positive result
                item[2] = (item[2] + noise_val) % self.modulus

    def detach_blinding_shares(self):
        shares = self.shares
        # TODO: secure delete
        del self.shares
        self.shares = None
        for uid in shares:
            shares[uid]['secret'] = b64encode(shares[uid]['secret'])
            shares[uid]['hash_id'] = b64encode(shares[uid]['hash_id'])
        return shares

    def import_blinding_share(self, share):
        '''
        reverse blinding factors for all of the counters
        '''
        self._unblind(b64decode(share['secret']))

    def increment(self, counter_key, bin_value, num_increments=1L):
        if self.counters is not None and counter_key in self.counters:
            for item in self.counters[counter_key]['bins']:
                if bin_value >= item[0] and bin_value < item[1]:
                    item[2] = (item[2] + long(num_increments)) % self.modulus

    def _tally_counter(self, counter):
        if self.counters == None:
            return False

        # validate that the counters match
        for key in self.counters:
            if key not in counter:
                return False
            if 'bins' not in counter[key]:
                return False
            num_bins = len(self.counters[key]['bins'])
            if num_bins != len(counter[key]['bins']):
                return False
            for i in xrange(num_bins):
                tally_item = counter[key]['bins'][i]
                if len(tally_item) != 3:
                    return False

        # ok, the counters match
        for key in self.counters:
            num_bins = len(self.counters[key]['bins'])
            for i in xrange(num_bins):
                tally_bin = self.counters[key]['bins'][i]
                tally_bin[2] = (tally_bin[2] + counter[key]['bins'][i][2]) % self.modulus

        # success
        return True

    def tally_counters(self, counters):
        # first add up all of the counters together
        for counter in counters:
            if not self._tally_counter(counter):
                return False
        # now adjust so our tally can register negative counts
        # (negative counts are possible if noise is negative)
        for key in self.counters:
            for tally_bin in self.counters[key]['bins']:
                tally_bin[2] = adjust_count_signed(tally_bin[2], self.modulus)
        return True

    def detach_counts(self):
        counts = self.counters
        self.counters = None
        return counts

"""
def prob_exit(consensus_path, my_fingerprint, fingerprint_pool=None):
    '''
    this func is currently unused
    if it becomes used later, we must add stem as a required python library
    '''
    from stem.descriptor import parse_file

    if fingerprint_pool == None:
        fingerprint_pool = [my_fingerprint]

    net_status = next(parse_file(consensus_path, document_handler='DOCUMENT', validate=False))
    DW = float(net_status.bandwidth_weights['Wed'])/10000
    EW = float(net_status.bandwidth_weights['Wee'])/10000

    # we must use longs here, because otherwise sum_of_sq_bw can overflow on
    # platforms where python has 32-bit ints
    # (on these platforms, this happens when router_entry.bandwidth > 65535)
    my_bandwidth, DBW, EBW, sum_of_sq_bw = 0L, 0L, 0L, 0L

    if my_fingerprint in net_status.routers:
        my_bandwidth = net_status.routers[my_fingerprint].bandwidth

    for (fingerprint, router_entry) in net_status.routers.items():
        if fingerprint not in fingerprint_pool or 'BadExit' in router_entry.flags:
            continue

        if 'Guard' in router_entry.flags and 'Exit' in router_entry.flags:
            DBW += router_entry.bandwidth
            sum_of_sq_bw += router_entry.bandwidth**2

        elif 'Exit' in router_entry.flags:
            EBW += router_entry.bandwidth
            sum_of_sq_bw += router_entry.bandwidth**2

    TEWBW = DBW*DW + EBW*EW
    prob = my_bandwidth/TEWBW
    sum_of_sq = sum_of_sq_bw/(TEWBW**2)
    return prob, sum_of_sq
"""

class TrafficModel(object):

    def __init__(self, states, start_p, trans_p, emit_p):
        self.states = states
        self.start_p = start_p
        self.trans_p = trans_p
        self.emit_p = emit_p

    def get_counter_labels(self):
        labels = []
        for state in self.emit_p:
            for direction in self.emit_p[state]:
                labels.append("TrafficModelTotalEmissions_{}_{}".format(state, direction))
                labels.append("TrafficModelTotalDelay_{}_{}".format(state, direction))
        for src_state in self.trans_p:
            for dst_state in self.trans_p[src_state]:
                labels.append("TrafficModelTotalTransitions_{}_{}".format(src_state, dst_state))
        return labels

    def run_viterbi(self, obs):
        V = [{}]
        for st in self.states:
            # updated emit_p here
            (direction, delay) = obs[0]
            (dp, dlam) = self.emit_p[st][direction]
            fitprob = log(dp) + log(dlam) - (delay*dlam)
            # replaced the following line
            #V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
            V[0][st] = {"prob": log(self.start_p[st]) + fitprob, "prev": None}
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in self.states:
                max_tr_prob = max(V[t-1][prev_st]["prob"]+log(self.trans_p[prev_st][st]) for prev_st in self.states)
                for prev_st in self.states:
                    if V[t-1][prev_st]["prob"] + log(self.trans_p[prev_st][st]) == max_tr_prob:
                        # updated emit_p here
                        (direction, delay) = obs[t]
                        (dp, dlam) = self.emit_p[st][direction]
                        fitprob = log(dp) + log(dlam) - (delay*dlam)
                        # replaced the following line
                        #max_prob = max_tr_prob * emit_p[st][obs[t]]
                        max_prob = max_tr_prob + fitprob
                        V[t][st] = {"prob": max_prob, "prev": prev_st}
                        break
        #for line in dptable(V):
        #    print line
        opt = []
        # The highest probability
        max_prob = max(value["prob"] for value in V[-1].values())
        previous = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] == max_prob:
                opt.append(st)
                previous = st
                break
        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        #print 'The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob
        return opt # list of highest probable states, in order
