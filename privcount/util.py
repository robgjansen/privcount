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
from os import urandom
from math import sqrt, exp, log
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

def log_error():
    _, _, tb = sys.exc_info()
    #traceback.print_tb(tb) # Fixed format
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]
    logging.warning("An error occurred in file '%s', at line %d, in func %s, in statement '%s'", filename, line, func, text)

def noise(sigma, sum_of_sq, p_exit):
    sigma_i = p_exit * sigma / sqrt(sum_of_sq)
    random_sample = gauss(0, sigma_i)
    return random_sample

def PRF(key, IV):
    return Hash("PRF1|KEY:%s|IV:%s|" % (key, IV)).digest()

def sample(s, q):
    ## Unbiased sampling through rejection sampling
    while True:
        v = struct.unpack("<L", s[:4])[0]
        if 0 <= v < q:
            break
        s = Hash(s).digest()
    return v

def derive_blinding_factor(label, secret, q, positive=True):
    ## Keyed share derivation
    s = PRF(secret, label)
    v = sample(s, q)
    s0 = v if positive else q - v
    return s0

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

    see privcount/test/test_counters.py for a test case
    '''

    def __init__(self, counters, q):
        self.counters = deepcopy(counters)
        self.q = q
        self.shares = None

        # initialize all counters to 0
        for key in self.counters:
            if 'bins' not in self.counters[key]:
                return None
            for item in self.counters[key]['bins']:
                assert len(item) == 2
                item.append(0.0) # bin is now, e.g.: [0.0, 512.0, 0.0] for bin_left, bin_right, count

    def _derive_all_counters(self, secret, positive):
        for key in self.counters:
            for item in self.counters[key]['bins']:
                label = "{}_{}_{}".format(key, item[0], item[1])
                blinding_factor = derive_blinding_factor(label, secret, self.q, positive=positive)
                item[2] = (item[2] + blinding_factor) % self.q

    def _blind(self, secret):
        self._derive_all_counters(secret, True)

    def _unblind(self, secret):
        self._derive_all_counters(secret, False)

    def generate(self, uids, noise_weight):
        self.shares = {}
        for uid in uids:
            secret = urandom(20)
            hash_id = PRF(secret, "KEYID")
            self.shares[uid] = {'secret': secret, 'hash_id': hash_id}
            # add blinding factors to all of the counters
            self._blind(secret)

	    # Add noise for each counter independently
        for key in self.counters:
            for item in self.counters[key]['bins']:
                sigma = self.counters[key]['sigma']
                sampled_noise = noise(sigma, 1, noise_weight)
                noise_val = int(round(sampled_noise))
                item[2] = (item[2] + noise_val) % self.q

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
        # reverse blinding factors for all of the counters
        self._unblind(b64decode(share['secret']))

    def increment(self, counter_key, bin_value, num_increments=1.0):
        if self.counters is not None and counter_key in self.counters:
            for item in self.counters[counter_key]['bins']:
                if bin_value >= item[0] and bin_value < item[1]:
                    item[2] = (item[2] + num_increments) % self.q

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
                tally_bin[2] = (tally_bin[2] + counter[key]['bins'][i][2]) % self.q

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
                if tally_bin[2] > (self.q / 2):
                    tally_bin[2] -= self.q
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
    EW = float(net_status.bandwidth_weights['Wed'])/10000

    my_bandwidth, DBW, EBW, sum_of_sq_bw = 0, 0, 0, 0

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
                labels.append("TrafficModelTotalEmissions_{}{}".format(state, direction))
                labels.append("TrafficModelTotalDelay_{}{}".format(state, direction))
        for src_state in self.trans_p:
            for dst_state in self.trans_p[src_state]:
                labels.append("TrafficModelTotalTransitions_{}{}".format(src_state, dst_state))
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
