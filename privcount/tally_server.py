'''
Created on Dec 12, 2015

@author: rob

See LICENSE for licensing information
'''
import os
import json
import logging
import cPickle as pickle
import yaml

from time import time
from copy import copy, deepcopy
from base64 import b64encode

from twisted.internet import reactor, task, ssl
from twisted.internet.protocol import ServerFactory

from privcount.config import normalise_path, choose_secret_handshake_path, _extra_keys, _common_keys
from privcount.counter import SecureCounters, counter_modulus, min_blinded_counter_value, max_blinded_counter_value, min_tally_counter_value, max_tally_counter_value, add_counter_limits_to_config, check_noise_weight_config, check_counters_config, CollectionDelay, float_accuracy, count_bins, are_events_expected
from privcount.crypto import generate_keypair, generate_cert
from privcount.log import log_error, format_elapsed_time_since, format_elapsed_time_wait, format_delay_time_until, format_interval_time_between, format_last_event_time_since, errorCallback, summarise_string, summarise_list
from privcount.match import exact_match_prepare_collection, suffix_match_prepare_collection, ipasn_prefix_match_prepare_string, load_match_list, load_as_prefix_map, exact_match, suffix_match, suffix_match_validate_item, exact_match_validate_item
from privcount.node import PrivCountNode, PrivCountServer, continue_collecting, log_tally_server_status, EXPECTED_EVENT_INTERVAL_MAX, EXPECTED_CONTROL_ESTABLISH_MAX
from privcount.protocol import PrivCountServerProtocol, get_privcount_version
from privcount.statistics_noise import get_noise_allocation, get_sanity_check_counter, DEFAULT_DUMMY_COUNTER_NAME
from privcount.traffic_model import TrafficModel, check_traffic_model_config

# for warning about logging function and format # pylint: disable=W1202
# for calling methods on reactor # pylint: disable=E1101

class TallyServer(ServerFactory, PrivCountServer):
    '''
    receive blinded counts from the DCs
    receive key shares from the SKs
    sum shares and counts at end of epoch
    publish the final results to a file
    '''

    def __init__(self, config_filepath):
        PrivCountServer.__init__(self, config_filepath)
        self.clients = {}
        self.collection_phase = None
        self.idle_time = time()
        self.num_completed_collection_phases = 0
        self.refresh_task = None

    def buildProtocol(self, addr):
        '''
        Called by twisted
        '''
        return PrivCountServerProtocol(self)

    def startFactory(self):
        '''
        Called by twisted
        '''
        # TODO
        return
        state = self.load_state()
        if state is not None:
            self.clients = state['clients']
            self.collection_phase = state['collection_phase']
            self.idle_time = state['idle_time']

    def stopFactory(self):
        # TODO
        if self.refresh_task is not None and self.refresh_task.running:
            self.refresh_task.stop()
            self.refresh_task = None
        return
        if self.collection_phase is not None or len(self.clients) > 0:
            # export everything that would be needed to survive an app restart
            state = {'clients': self.clients, 'collection_phase': self.collection_phase, 'idle_time': self.idle_time}
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

        # refresh and check status every event_period seconds
        self.refresh_task = task.LoopingCall(self.refresh_loop)
        refresh_deferred = self.refresh_task.start(self.config['event_period'], now=False)
        refresh_deferred.addErrback(errorCallback)

        # setup server for receiving blinded counts from the DC nodes and key shares from the SK nodes
        listen_port = self.config['listen_port']
        key_path = self.config['key']
        cert_path = self.config['cert']
        ssl_context = ssl.DefaultOpenSSLContextFactory(key_path, cert_path)

        logging.info("Tally Server listening on port {}".format(listen_port))
        reactor.listenSSL(listen_port, self, ssl_context)
        reactor.run()

    def refresh_loop(self):
        '''
        Perform the TS event loop:
        Refresh the config, check clients, check if we want to start or stop
        collecting, and log a status update.
        This function is called using LoopingCall, so any exceptions will be
        turned into log messages. (This is the behaviour we want for malformed
        config files.)
        '''
        # make sure we have the latest config and counters
        self.refresh_config()

        # check if any clients have not checked in recently
        self.clear_dead_clients()

        # check if we should start the next collection phase
        if self.collection_phase is None:
            num_phases = self.num_completed_collection_phases
            if continue_collecting(num_phases,
                                   self.config['continue'],
                                   'idle'):
                dcs, sks = self.get_idle_dcs(), self.get_idle_sks()
                if len(dcs) >= self.config['dc_threshold'] and len(sks) >= self.config['sk_threshold']:
                    if self.collection_delay.round_start_permitted(
                            self.config['noise'],
                            time(),
                            self.config['delay_period'],
                            max_client_rtt=self.get_max_all_client_rtt(),
                            always_delay=self.config['always_delay'],
                            tolerance=self.config['sigma_decrease_tolerance']):
                        # we've passed all the checks, start the collection
                        num_phases = self.num_completed_collection_phases
                        logging.info("starting collection phase {} with {} DataCollectors and {} ShareKeepers".format((num_phases+1), len(dcs), len(sks)))
                        self.start_new_collection_phase(dcs, sks)

        # check if we should stop a running collection phase
        else:
            if self.collection_phase.is_error():
                logging.info("stopping collection phase due to error")
                self.stop_collection_phase()

            elif self.collection_phase.is_expired():
                logging.info("stopping collection phase due to valid expiration")
                self.stop_collection_phase()

        # log the latest status
        log_tally_server_status(self.get_status())
        if self.collection_phase is not None:
            self.collection_phase.log_status()

    @staticmethod
    def load_match_file(config,
                        file_path,
                        old_match_files=None,
                        old_match_lists=None,
                        new_match_lists=None,
                        check_domain=False,
                        check_country=False,
                        check_as=False,
                        check_reason=False,
                        check_onion=False):
        '''
        Load a match file from file_path.

        If new_match_lists is not none, append the file's contents to that
        list as an array.

        If check_domain, then check that each line in the file is a
        potentially valid domain name, and lowecase the string.
        If check_country, then check that each line in the file is a
        potentially valid MaxMind Country Code, and lowercase the string.
        If check_as, then check that each line in the file is a potentially
        valid CAIDA AS Number, and load the first AS number in each AS set or
        multi-origin AS.
        If check_reason, then check that each line in the file is a
        potentially valid HSDir Fetch, HSDir Store, or Circuit Failure reason,
        and lowercase the string.
        If check_onion, then check that each line in the file is a
        potentially valid onion address, after stripping non-onion address
        components from URLs or domains, and lowercasing the resulting string.

        If config is None, or the file is not in old_match_files, or the
        contents do not match old_match_lists, then mark the list as changed.

        Returns a tuple containing the normalised file_path, and a boolean
        indicating whether the list changed since the last time it was loaded.
        '''
        assert file_path is not None
        (file_path, match_list) = load_match_list(file_path,
                                                  check_domain=check_domain,
                                                  check_country=check_country,
                                                  check_as=check_as,
                                                  check_reason=check_reason,
                                                  check_onion=check_onion)

        # lists must have at least one entry
        assert len(match_list) > 0

        # it takes about 5 seconds to process the Alexa Top 1 million
        # so let the caller know if the list changed
        has_list_changed = False
        if (config is None or
            old_match_files is None or
            file_path not in old_match_files or
            old_match_lists is None or
            match_list != old_match_lists[old_match_files.index(file_path)]):
            has_list_changed = True
            logging.info("Changed list: '{}'".format(file_path))


        # Now, add the raw list to the list of match lists
        if new_match_lists is not None:
            new_match_lists.append(match_list)

        return (file_path, has_list_changed)

    @staticmethod
    def process_match_list(match_list,
                           prepare_exact=False,
                           prepare_suffix=False,
                           suffix_separator=None,
                           existing_exacts=None,
                           existing_suffixes=None,
                           collection_tag=-1):
        '''
        Process a match list from match_list.

        If existing_exacts or existing_suffixes are not None, process the
        match list's contents into that data structure.

        If prepare_exact is True, check that the list can be prepared for
        exact matching. If existing_exacts is not None, add the exacts
        to existing_exacts.

        If prepare_suffix is True, check that the list can be prepared for
        suffix matching. If existing_suffixes is not None, add the suffixes
        to existing_suffixes with tag collection_tag.
        '''
        assert match_list is not None
        # lists must have at least one entry
        assert len(match_list) > 0

        if prepare_exact or prepare_suffix:
            logging.info("Checking matches for '{}'"
                         .format(summarise_list(match_list,
                                                sort_output=False)))
        # make sure the DCs will be able to process the lists
        if prepare_exact:
            exact_match_prepare_collection(match_list,
                                           existing_exacts=existing_exacts)
        if prepare_suffix:
            suffix_match_prepare_collection(match_list,
                                            separator=suffix_separator,
                                            existing_suffixes=existing_suffixes,
                                            collection_tag=collection_tag)

    @staticmethod
    def load_as_prefix_file(config, ip_version, file_path, new_as_prefix_maps,
                            old_as_prefix_files, old_as_prefix_maps,
                            prepare_prefix=True):
        '''
        Load an AS prefix file containing tab-separated IPv4 or IPv6 network
        prefixes and CAIDA AS numbers. Each network mapping is separated by a
        newline. Loads data from from file_path into
        new_as_prefix_maps[ip_version].

        If config is None, or file_path is not equal to
        old_as_prefix_files[ip_version], or the contents do not match
        old_as_prefix_maps[ip_version], then check that the list can be
        prepared for matching, based on prepare_prefix.

        Returns the normalised file_path, and a boolean indicating whether the
        list changed since the last time it was loaded.
        '''
        assert ip_version == 4 or ip_version == 6

        has_list_changed = False

        # import this list of address / prefix / AS number lines
        # This takes under a second for the coalesced IPv4 prefixes
        (file_path, map_list) = load_as_prefix_map(file_path)

        # maps must not be empty
        assert len(map_list) > 0

        # just send the file to the DCs as a string
        prefix_map = "\n".join(map_list)

        # only re-process the map when it changes
        if (config is None or
            file_path != old_as_prefix_files.get(ip_version) or
            prefix_map != old_as_prefix_maps.get(ip_version)):
            has_list_changed = True
            if prepare_prefix:
                # make sure the DCs will be able to process the lists
                logging.info("Checking IPv{} prefixes in '{}'"
                             .format(ip_version, file_path))
                # check that the format is parseable before sending to DCs
                prefix_map_obj = ipasn_prefix_match_prepare_string(
                                     prefix_map)

        # Now, add the map to the dict of prefix maps
        new_as_prefix_maps[ip_version] = prefix_map
        return (file_path, has_list_changed)

    @staticmethod
    def modify_bins(bin_count, counter, add_inf_bin=True):
        '''
        Put bin_count bins in counter, with ranges:
            [0, 1), [1, 2), ... , [bin_count-1, bin_count)
        If add_inf_bin is True, also add a bin with range:
            [bin_count, inf]
        '''
        assert bin_count > 0
        # Don't assume there are any existing bins
        counter['bins'] = []
        counter_bins = counter['bins']
        # Add bins up to bin_count
        for bin in xrange(bin_count):
            counter_bins.append([float(bin), float(bin+1)])
        if add_inf_bin:
            counter_bins.append([float(bin_count), float('inf')])

    @staticmethod
    def modify_counter_list_bins(bin_count, counter_list, add_inf_bin=True,
                                 counter_filter=None):
        '''
        Use modify_bins(bin_count, add_inf_bin=add_inf_bin) on each counter in
        counter_list.

        Uses counter_filter to find the counter names to modify.
        '''
        assert counter_list is not None
        for counter_name in counter_list:
            if counter_filter is not None and counter_filter(counter_name):
                TallyServer.modify_bins(bin_count,
                                        counter_list[counter_name])

    @staticmethod
    def load_match_config(old_config,
                          new_config,
                          files_key,
                          lists_key,
                          check_domain=False,
                          check_country=False,
                          check_as=False,
                          check_reason=False,
                          check_onion=False,
                          counters_key='counters',
                          counter_filter=None,
                          old_processed_config=None,
                          new_processed_config=None,
                          exacts_key=None,
                          suffixes_key=None,
                          suffix_separator=None,
                          validate=True,
                          reject_overlapping_lists=False):
        '''
        Load raw list data from each file in new_config[files_key] into
        new_config[lists_key], using old_config to check if the files need
        to be re-processed. Update the paths in new_config[files_key] to
        absolute paths.

        Pass check_* directly to load_match_file().

        Once the files have been processed, update the bins for counters in
        counters_key that return true for counter_filter(counter_name).

        If old_processed_config is not None, use it instead of old_config for
        exacts_key and suffixes_key. If new_processed_config is not None,
        use it in a similar way.

        If exacts_key is not None, process the lists as exact match lists into
        exacts_key. If suffixes_key is not None, process the lists as suffix
        match lists into suffixes_key, using suffix_separator.

        If reject_overlapping_lists is True, assert if any of the lists are
        overlapping. Otherwise, warn and remove overlaps.
        '''
        assert new_config is not None
        assert files_key is not None
        assert lists_key is not None
        assert counter_filter is not None
        # do we ever  you want to load a list without processing it?
        assert exacts_key is not None or suffixes_key is not None
        assert (suffixes_key is None) == (suffix_separator is None)

        # set defaults if missing
        if old_processed_config is None:
            old_processed_config = old_config
        if new_processed_config is None:
          new_processed_config = new_config

        old_match_files = old_config.get(files_key, []) if old_config is not None else []
        old_match_lists = old_config.get(lists_key, []) if old_config is not None else []
        old_match_exacts = old_processed_config.get(exacts_key, {}) if (old_processed_config is not None and exacts_key is not None) else []
        old_match_suffixes = old_processed_config.get(suffixes_key, {}) if (old_processed_config is not None and suffixes_key is not None) else {}
        new_config[lists_key] = []
        if exacts_key is not None:
            new_processed_config[exacts_key] = []
        if suffixes_key is not None:
            new_processed_config[suffixes_key] = {}
        if files_key in new_config and len(new_config[files_key]) > 0:
            # Track list changes for processed list reloads
            has_any_previous_list_changed = False
            # reload all the raw lists
            for i in xrange(len(new_config[files_key])):
                (file_path,
                 has_list_changed) = TallyServer.load_match_file(
                                         old_config,
                                         new_config[files_key][i],
                                         old_match_files=old_match_files,
                                         old_match_lists=old_match_lists,
                                         new_match_lists=new_config[lists_key],
                                         check_domain=check_domain,
                                         check_country=check_country,
                                         check_as=check_as,
                                         check_reason=check_reason,
                                         check_onion=check_onion)
                new_config[files_key][i] = file_path
                if has_list_changed:
                    has_any_previous_list_changed = True
            if not has_any_previous_list_changed:
                # skip processing the lists into new_processed_config,
                # because they have not changed
                if exacts_key is not None:
                    new_processed_config[exacts_key] = old_match_exacts
                if suffixes_key is not None:
                    new_processed_config[suffixes_key] = old_match_suffixes
            else:
              # process the lists
              for i in xrange(len(new_config[files_key])):
                  if exacts_key is not None:
                      TallyServer.process_match_list(
                          new_config[lists_key][i],
                          prepare_exact=True,
                          existing_exacts=new_processed_config[exacts_key])
                  if suffixes_key is not None:
                      TallyServer.process_match_list(
                          new_config[lists_key][i],
                          prepare_suffix=True,
                          suffix_separator=suffix_separator,
                          existing_suffixes=new_processed_config[suffixes_key],
                          collection_tag=i)
            # modify the bins
            TallyServer.modify_counter_list_bins(
               len(new_config[files_key]),
               new_config[counters_key],
               counter_filter=counter_filter)
            # and check that the matching actually works
            if validate:
                for i in xrange(len(new_config[lists_key])):
                    for item in new_config[lists_key][i]:
                        if exacts_key is not None:
                            # if there are any duplicates, we can only validate
                            # the first list
                            if reject_overlapping_lists or i == 0:
                                exact_match_validate_item(new_processed_config[exacts_key][i],
                                                          item,
                                                          new_config[lists_key][i])
                        if suffixes_key is not None:
                            suffix_match_validate_item(new_processed_config[suffixes_key],
                                                       item,
                                                       new_config[lists_key][i],
                                                       separator=suffix_separator,
                                                       expected_collection_tag=i,
                                                       reject_overlapping_lists=reject_overlapping_lists)

        logging.debug("Counts: Files: {} Raw: {} Exacts: {} Suffixes: {}"
                      .format(len(new_config.get(files_key, [])),
                              len(new_config[lists_key]),
                              len(new_processed_config.get(exacts_key, [])),
                              len(new_processed_config.get(suffixes_key, []))))
        assert len(new_config.get(files_key, [])) == len(new_config[lists_key])
        if exacts_key is not None:
            assert len(new_config.get(files_key, [])) == len(new_processed_config[exacts_key])
        if suffixes_key is not None:
            assert (len(new_config.get(files_key, [])) > 0) == (len(new_processed_config[suffixes_key]) > 0)

    @staticmethod
    def load_as_prefix_config(old_config,
                              new_config,
                              files_key,
                              old_processed_config=None,
                              new_processed_config=None,
                              maps_key=None):
        '''
        Load raw AS prefix data from each file in new_config[files_key] into
        new_config[maps_key], using old_config to check if the files need to
        be re-processed. Update the paths in new_config[files_key] to absolute
        paths.

        If old_processed_config is not None, use it instead of old_config for
        maps_key and suffixes_key. If new_processed_config is not None,
        use it in a similar way.

        Process the lists as AS prefix lists to confirm they are valid, then
        discard the results.
        '''
        assert new_config is not None
        assert files_key is not None
        assert maps_key is not None

        # set defaults if missing
        if old_processed_config is None:
            old_processed_config = old_config
        if new_processed_config is None:
          new_processed_config = new_config

        # optional mappings of IPv4 and IPv6 prefixes to CAIDA AS numbers
        # Custom code for loading the prefix maps unconditionally
        old_as_prefix_files = old_config.get(files_key, {}) if old_config is not None else {}
        old_as_prefix_maps = old_processed_config.get(maps_key, {})
        new_processed_config[maps_key] = {}
        # you must have both IPv4 and IPv6 prefix mappings, or neither
        # we keep IPv4 and IPv6 prefixes separate because it makes matching faster
        if (files_key in new_config and len(new_config[files_key]) == 2):
            # always reload all the prefix maps
            for ipv in new_config[files_key]:
                assert ipv == 4 or ipv == 6
                (file_path, _) = TallyServer.load_as_prefix_file(
                                     old_config,
                                     ipv,
                                     new_config[files_key][ipv],
                                     new_processed_config[maps_key],
                                     old_as_prefix_files,
                                     old_as_prefix_maps,
                                     prepare_prefix=True)
                new_config[files_key][ipv] = file_path

        assert len(new_config.get(files_key, [])) == len(new_processed_config[maps_key])
        # you must have both IPv4 and IPv6 prefix mappings, or neither
        assert (4 in new_processed_config[maps_key]) == (6 in new_processed_config[maps_key])


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
            ts_conf = conf['tally_server']

            # a private/public key pair and a cert containing the public key
            # if either path is not specified, use the default path
            if 'key' in ts_conf and 'cert' in ts_conf:
                ts_conf['key'] = normalise_path(ts_conf['key'])
                ts_conf['cert'] = normalise_path(ts_conf['cert'])
            else:
                ts_conf['key'] = normalise_path('privcount.rsa_key.pem')
                ts_conf['cert'] = normalise_path('privcount.rsa_key.cert')
            # generate a new key and cert if either file does not exist
            if (not os.path.exists(ts_conf['key']) or
                not os.path.exists(ts_conf['cert'])):
                generate_keypair(ts_conf['key'])
                generate_cert(ts_conf['key'], ts_conf['cert'])

            # find the path for the secret handshake file
            ts_conf['secret_handshake'] = choose_secret_handshake_path(
                ts_conf, conf)
            # check we can load the secret handshake file, creating if needed
            # (but ignore the actual secret, it never forms part of our config)
            # we can't use PrivCountProtocol.handshake_secret(), because
            # our own self.config is either None or outdated at this point
            assert PrivCountServerProtocol.handshake_secret_load(
                ts_conf['secret_handshake'],
                create=True)

            ts_conf.setdefault('circuit_sample_rate', 1.0)
            ts_conf['circuit_sample_rate'] = \
                float(ts_conf['circuit_sample_rate'])
            assert ts_conf['circuit_sample_rate'] >= 0.0
            assert ts_conf['circuit_sample_rate'] <= 1.0

            ts_conf.setdefault('max_cell_events_per_circuit', -1)
            ts_conf['max_cell_events_per_circuit'] = \
                int(ts_conf['max_cell_events_per_circuit'])

            # the counter bin file
            if 'counters' in ts_conf:
                ts_conf['counters'] = normalise_path(ts_conf['counters'])
                assert os.path.exists(ts_conf['counters'])
                with open(ts_conf['counters'], 'r') as fin:
                    counters_conf = yaml.load(fin)
                ts_conf['counters'] = counters_conf['counters']
            else:
                ts_conf['counters'] = conf['counters']

            # the counter noise config - one of the following must be provided:
            # noise: contains the noise allocation parameters,
            # sigmas: contains pre-calculated sigmas
            # (if both are provided, sigmas is ignored)
            # noise file
            if 'noise' in ts_conf:
                ts_conf['noise'] = normalise_path(ts_conf['noise'])
                assert os.path.exists(ts_conf['noise'])
                with open(ts_conf['noise'], 'r') as fin:
                    noise_conf = yaml.load(fin)
                # use both the privacy and counters elements from noise_conf
                ts_conf['noise'] = {}
                ts_conf['noise']['privacy'] = noise_conf['privacy']
                ts_conf['noise']['counters'] = noise_conf['counters']
            # noise config in the same file
            elif 'privacy' in conf and 'counters' in conf:
                ts_conf['noise'] = {}
                ts_conf['noise']['privacy'] = conf['privacy']
                ts_conf['noise']['counters'] = conf['counters']
            # sigmas file
            elif 'sigmas' in ts_conf:
                ts_conf['sigmas'] = normalise_path(ts_conf['sigmas'])
                assert os.path.exists(ts_conf['sigmas'])
                with open(ts_conf['sigmas'], 'r') as fin:
                    sigmas_conf = yaml.load(fin)
                ts_conf['noise'] = {}
                ts_conf['noise']['counters'] = sigmas_conf['counters']
                # we've packed it into ts_conf['noise'], so remove it
                del ts_conf['sigmas']
            # sigmas config in the same file
            else:
                ts_conf['noise'] = {}
                ts_conf['noise']['counters'] = conf['counters']

            # if we are counting a traffic model
            if 'traffic_model' in ts_conf:
                # we need the model, which specifies which counters we need to count
                # make sure the model file exists
                ts_conf['traffic_model'] = normalise_path(ts_conf['traffic_model'])
                assert os.path.exists(ts_conf['traffic_model'])

                # import and validate the model
                with open(ts_conf['traffic_model'], 'r') as fin:
                    traffic_model_conf = json.load(fin)
                    assert check_traffic_model_config(traffic_model_conf)

                # store the configs so we can transfer them later
                ts_conf['traffic_model'] = traffic_model_conf

                # get an object and register the dynamic counters
                tmodel = TrafficModel(traffic_model_conf)

                # we also need noise parameters for all of the traffic model counters
                # make sure the noise file exists
                traffic_noise_conf = None
                if 'traffic_noise' in ts_conf:
                    ts_conf['traffic_noise'] = normalise_path(ts_conf['traffic_noise'])
                    assert os.path.exists(ts_conf['traffic_noise'])

                    # import and validate the noise
                    with open(ts_conf['traffic_noise'], 'r') as fin:
                        traffic_noise_conf = yaml.load(fin)
                    assert tmodel.check_noise_config(traffic_noise_conf)

                    # store the configs so we can transfer them later
                    ts_conf['traffic_noise'] = traffic_noise_conf

                # supplying a traffic model implies that the tally server
                # wants to enable all counters associated with that model
                # register the dynamic counter labels that will be needed for this model
                tmodel.register_counters()

                # get the bins and noise that we should use for this model
                tmodel_bins = tmodel.get_bins_init_config(traffic_noise_conf)

                if 'traffic_noise' in ts_conf:
                    tmodel_noise = tmodel.get_noise_init_config(traffic_noise_conf)

                    # sanity check
                    if set(tmodel_bins.keys()) != set(tmodel_noise.keys()):
                        logging.error("the set of initial bins and noise labels are not equal")
                        assert set(tmodel_bins.keys()) != set(tmodel_noise.keys())

                # inject the traffic model counter bins and noise configs, i.e.,
                # append the traffic model bins and noise to the other configured values
                ts_conf['counters'].update(tmodel_bins)
                if 'traffic_noise' in ts_conf:
                    ts_conf['noise']['counters'].update(tmodel_noise)
                if set(ts_conf['counters'].keys()) != set(ts_conf['noise']['counters'].keys()):
                    logging.error("the set of traffic model bins and noise labels are not equal")
                    assert set(ts_conf['counters'].keys()) != set(ts_conf['noise']['counters'].keys())

            # Do we reject overlapping lists, or warn and remove overlaps?
            ts_conf.setdefault('reject_overlapping_lists', True)
            assert isinstance(ts_conf['reject_overlapping_lists'], bool)

            # optional lists and processed suffixes of DNS domain names
            TallyServer.load_match_config(
                self.config,
                ts_conf,
                'domain_files',
                'domain_lists',
                check_domain=True,
                # Domains only have Exit counters
                counter_filter=(lambda counter_name: counter_name.startswith("ExitDomain") and
                                                     counter_name.endswith("CountList")),
                exacts_key='domain_exacts',
                suffixes_key='domain_suffixes',
                suffix_separator=".",
                reject_overlapping_lists=ts_conf['reject_overlapping_lists'])

            # optional lists of country codes from the MaxMind GeoIP database
            TallyServer.load_match_config(
                self.config,
                ts_conf,
                'country_files',
                'country_lists',
                check_country=True,
                # Countries have both Entry and NonEntry Connection counters
                counter_filter=(lambda counter_name: "CountryMatch" in counter_name and
                                                     counter_name.endswith("CountList")),
                exacts_key='country_exacts',
                reject_overlapping_lists=ts_conf['reject_overlapping_lists'])

            # as_data is used as the processed config by the prefix maps and the AS lists
            old_as_data = self.config.get('as_data', {}) if self.config is not None else {}
            ts_conf['as_data'] = {}

            # optional mappings of IPv4 and IPv6 prefixes to CAIDA AS numbers
            TallyServer.load_as_prefix_config(
                self.config,
                ts_conf,
                'as_prefix_files',
                # we put the raw prefixes in as_data to make them easier to
                # send to the data collectors
                old_processed_config=old_as_data,
                new_processed_config=ts_conf['as_data'],
                maps_key='prefix_maps')

            # optional lists of AS numbers from the CAIDA AS prefix files or AS rankings
            TallyServer.load_match_config(
                self.config,
                ts_conf,
                'as_files',
                # the raw lists have a different name to avoid confusion with
                # ['as_data]['lists'], which has to keep the same name for
                # backwards compatibility
                'as_raw_lists',
                check_as=True,
                # ASs have both Entry and NonEntry Connection counters
                counter_filter=(lambda counter_name: "ASMatch" in counter_name and
                                                     counter_name.endswith("CountList")),
                # we put the processed lists in as_data to make them easier to
                # send to the data collectors
                old_processed_config=old_as_data,
                new_processed_config=ts_conf['as_data'],
                exacts_key='lists',
                reject_overlapping_lists=ts_conf['reject_overlapping_lists'])

            # do some additional checks on the AS lists
            # you must have both prefix mappings and AS lists, or neither
            assert (len(ts_conf['as_data']['prefix_maps']) > 0) == (len(ts_conf['as_data']['lists']) > 0)

            # optional lists of HSDir Store reasons
            TallyServer.load_match_config(
                self.config,
                ts_conf,
                'hsdir_store_files',
                'hsdir_store_lists',
                check_reason=True,
                # Store reason counters match HSDir*Store*ReasonCountList
                counter_filter=(lambda counter_name: counter_name.startswith("HSDir") and
                                                     "Store" in counter_name and
                                                     counter_name.endswith("ReasonCountList")),
                exacts_key='hsdir_store_exacts',
                reject_overlapping_lists=ts_conf['reject_overlapping_lists'])

            # optional lists of HSDir Fetch reasons
            TallyServer.load_match_config(
                self.config,
                ts_conf,
                'hsdir_fetch_files',
                'hsdir_fetch_lists',
                check_reason=True,
                # Fetch reason counters match HSDir*Fetch*ReasonCountList
                counter_filter=(lambda counter_name: counter_name.startswith("HSDir") and
                                                     "Fetch" in counter_name and
                                                     counter_name.endswith("ReasonCountList")),
                exacts_key='hsdir_fetch_exacts',
                reject_overlapping_lists=ts_conf['reject_overlapping_lists'])

            # optional lists of Circuit Failure reasons
            TallyServer.load_match_config(
                self.config,
                ts_conf,
                'circuit_failure_files',
                'circuit_failure_lists',
                check_reason=True,
                # Circuit Failure reason counters match *FailureCircuitReasonCountList
                counter_filter=(lambda counter_name: counter_name.endswith("FailureCircuitReasonCountList")),
                exacts_key='circuit_failure_exacts',
                reject_overlapping_lists=ts_conf['reject_overlapping_lists'])

            # optional lists of onion addresses for HSDir Store and Fetch events
            TallyServer.load_match_config(
                self.config,
                ts_conf,
                'onion_address_files',
                'onion_address_lists',
                check_onion=True,
                # Onion Address counters match HSDir*Store/Fetch*OnionAddressCountList
                counter_filter=(lambda counter_name: counter_name.startswith("HSDir") and
                                                     ("Store" in counter_name or "Fetch" in counter_name) and
                                                     counter_name.endswith("OnionAddressCountList")),
                exacts_key='onion_address_exacts',
                reject_overlapping_lists=ts_conf['reject_overlapping_lists'])


            # an optional noise allocation results file
            if 'allocation' in ts_conf:
                ts_conf['allocation'] = normalise_path(ts_conf['allocation'])
                assert os.path.exists(os.path.dirname(ts_conf['allocation']))

            # now all the files are loaded, use noise to calculate sigmas
            # (if noise was configured)
            if 'privacy' in ts_conf['noise']:
                ts_conf['noise'] = get_noise_allocation(
                    ts_conf['noise'],
                    circuit_sample_rate=ts_conf['circuit_sample_rate'])
                # and write it to the specified file (if configured)
                if 'allocation' in ts_conf:
                    with open(ts_conf['allocation'], 'w') as fout:
                        yaml.dump(ts_conf['noise'], fout,
                                  default_flow_style=False)

            # ensure we always add a sanity check counter
            ts_conf['counters'][DEFAULT_DUMMY_COUNTER_NAME] = get_sanity_check_counter()
            ts_conf['noise']['counters'][DEFAULT_DUMMY_COUNTER_NAME] = get_sanity_check_counter()

            # now we have bins and sigmas (and perhaps additional calculation
            # info along with the sigmas)
            # perform sanity checks, making sure all counter names are known
            # counters
            assert check_counters_config(ts_conf['counters'],
                                         ts_conf['noise']['counters'],
                                         allow_unknown_counters=False)

            # a directory for results files
            if 'results' in ts_conf:
                ts_conf['results'] = normalise_path(ts_conf['results'])
            else:
                ts_conf['results'] = normalise_path('./')
            assert os.path.exists(ts_conf['results'])

            # the state file (unused)
            ts_conf.pop('state', None)
            #ts_conf['state'] = normalise_path(ts_conf['state'])
            #assert os.path.exists(os.path.dirname(ts_conf['state']))

            # Must be configured manually
            assert 'collect_period' in ts_conf
            # Set the default periods
            ts_conf.setdefault('event_period', 600)
            ts_conf.setdefault('checkin_period', 600)

            # The event period should be less than or equal to half the
            # collect period, otherwise privcount sometimes takes an extra
            # event period to produce results
            event_max = ts_conf['collect_period']/2
            if (ts_conf['event_period'] > event_max):
                logging.warning("event_period %d too large for collect_period %d, reducing to %d",
                                ts_conf['event_period'],
                                ts_conf['collect_period'],
                                event_max)
                ts_conf['event_period'] = event_max

            # The checkin period must be less than or equal to half the
            # collect period, otherwise privcount never finishes.
            checkin_max = ts_conf['collect_period']/2
            if (ts_conf['checkin_period'] > checkin_max):
                logging.warning("checkin_period %d too large for collect_period %d, reducing to %d",
                                ts_conf['checkin_period'],
                                ts_conf['collect_period'],
                                checkin_max)
                ts_conf['checkin_period'] = checkin_max
            # It should also be less than or equal to the event period,
            # so that the TS is up to date with client statuses every
            # event loop.
            checkin_max_log = ts_conf['event_period']
            if (ts_conf['checkin_period'] > checkin_max_log):
                logging.info("checkin_period %d greater than event_period %d, client statuses might be delayed",
                             ts_conf['checkin_period'],
                             ts_conf['event_period'])

            ts_conf['delay_period'] = self.get_valid_delay_period(ts_conf)

            ts_conf.setdefault('always_delay', False)
            assert isinstance(ts_conf['always_delay'], bool)

            ts_conf['sigma_decrease_tolerance'] = \
                self.get_valid_sigma_decrease_tolerance(ts_conf)

            assert ts_conf['listen_port'] > 0
            assert ts_conf['sk_threshold'] > 0
            assert ts_conf['dc_threshold'] > 0
            assert ts_conf.has_key('noise_weight')
            assert check_noise_weight_config(ts_conf['noise_weight'],
                                             ts_conf['dc_threshold'])
            assert ts_conf['collect_period'] > 0
            assert ts_conf['event_period'] > 0
            assert ts_conf['checkin_period'] > 0
            # The TS runs one round by default
            ts_conf.setdefault('continue', False)
            assert (isinstance(ts_conf['continue'], bool) or
                    ts_conf['continue'] >= 0)
            # check the hard-coded counter values are sane
            assert counter_modulus() > 0
            assert min_blinded_counter_value() == 0
            assert max_blinded_counter_value() > 0
            assert max_blinded_counter_value() < counter_modulus()
            assert min_tally_counter_value() < 0
            assert max_tally_counter_value() > 0
            assert max_tally_counter_value() < counter_modulus()
            assert -min_tally_counter_value() < counter_modulus()

            if self.config == None:
                logging.info("using initial config = %s",
                             summarise_string(str(ts_conf)))
                logging.debug("using config (full value) = %s",
                              str(ts_conf))
            else:
                changed = False
                for k in _extra_keys(self.config, ts_conf):
                    changed = True
                    PrivCountNode.log_config_key_changed(k,
                                                         old_val_str=self.config[k])
                for k in _extra_keys(ts_conf, self.config):
                    changed = True
                    PrivCountNode.log_config_key_changed(k,
                                                         new_val_str=ts_conf[k])
                for k in _common_keys(self.config, ts_conf):
                    if self.config[k] != ts_conf[k]:
                        changed = True
                        PrivCountNode.log_config_key_changed(k,
                                                             old_val_str=self.config[k],
                                                             new_val_str=ts_conf[k])
                if not changed:
                    logging.debug('no config changes found')

            # unconditionally replace the config, even if it hasn't changed
            # this avoids bugs in the change logic above
            self.config = ts_conf

        except AssertionError:
            logging.warning("problem reading config file: invalid data")
            log_error()
        except KeyError:
            logging.warning("problem reading config file: missing required keys")
            log_error()

    MIN_SAFE_RTT = 2.0
    TYPICAL_RTT_JITTER = 1.0

    def get_max_client_rtt(self, uid):
        '''
        Get the maximum reasonable rtt for uid
        '''
        # Maximum RTT in ~2005 was 20 seconds
        # https://www3.cs.stonybrook.edu/~phillipa/papers/SPECTS.pdf
        # There's no guarantee the last rtt will be the same as this one,
        # so add a few seconds unconditionally
        return self.clients[uid].get('rtt', TallyServer.MIN_SAFE_RTT) + TallyServer.TYPICAL_RTT_JITTER

    def get_max_all_client_rtt(self):
        '''
        Get the maximum reasonable rtt for all clients
        '''
        max_rtts = [TallyServer.TYPICAL_RTT_JITTER]
        for uid in self.clients:
            max_rtts.append(self.get_max_client_rtt(uid))
        return max(max_rtts)

    def is_client_control_ok(self, uid):
        '''
        Has uid completed the control protocol with its tor instance within a
        reasonable amount of time, taking into account checkin period, rtt,
        collection phase start time, and clock padding?
        '''
        now = time()
        c_status = self.clients[uid]
        if c_status['type'] != 'DataCollector':
            return True

        # if the collection phase hasn't started, everything is ok
        if self.collection_phase is None:
            return True
        start_ts = self.collection_phase.get_start_ts()
        if start_ts is None:
            return True

        # if we've completed the control protocol, everything is ok
        if 'tor_privcount_version' in c_status:
            return True

        rtt = self.get_max_client_rtt(uid)
        clock_padding = self.collection_phase.clock_padding
        time_since_start = now - (start_ts + clock_padding)

        # This will also trigger if we miss a checkin at the start of the
        # round. That's ok.
        return time_since_start <= (EXPECTED_CONTROL_ESTABLISH_MAX +
                                    self.get_checkin_period() +
                                    rtt)

    def is_last_client_event_recent(self, uid):
        '''
        Is the last event from uid newer than EXPECTED_EVENT_INTERVAL_MAX,
        taking into account the checkin period, rtt, the collection phase
        start time, and clock padding?
        '''
        now = time()
        c_status = self.clients[uid]
        if c_status['type'] != 'DataCollector':
            return True

        # if the collection phase hasn't started, everything is ok
        if self.collection_phase is None:
            return True
        start_ts = self.collection_phase.get_start_ts()
        if start_ts is None:
            return True

        rtt = self.get_max_client_rtt(uid)
        clock_padding = self.collection_phase.clock_padding
        time_since_event = now - c_status.get('last_event_time',
                                              start_ts + clock_padding)

        return time_since_event <= (EXPECTED_EVENT_INTERVAL_MAX +
                                    2*self.get_checkin_period() +
                                    rtt)

    def are_dc_events_expected(self, uid, status=None):
        '''
        Return True if we expect the Data Collector at uid to receive events
        regularly.
        Return False if we don't, or if it's not a Data Collector.
        '''
        if self.get_client_type(uid, status) != 'DataCollector':
            return False

        flag_list = self.get_client_flag_list(uid, status)
        return are_events_expected(self.config['counters'], flag_list)

    def clear_dead_clients(self):
        '''
        Check how long it has been since clients have successfully contacted
        us, and mark clients that have been down for too long.
        Also warns if clients have been various kinds of down for a smaller
        amount of time.
        '''
        now = time()

        for uid in self.clients.keys():
            # don't print ShareKeepers' public keys, they're very long
            c_status = self.clients[uid].copy()
            if 'public_key' in c_status:
                c_status['public_key'] = "(public key)"
            time_since_checkin = now - c_status['alive']
            start_ts = None
            if self.collection_phase is not None:
                start_ts = self.collection_phase.get_start_ts()
            time_since_event = 0.0
            if c_status['type'] == 'DataCollector' and start_ts:
                time_since_event = now - c_status.get('last_event_time',
                                                      start_ts)
            rtt = self.get_max_client_rtt(uid)

            flag_message = "Is the relay in the Tor consensus?"
            flag_list = self.get_client_flag_list(uid)
            if flag_list is None:
                flag_message = ""
            elif len(flag_list) > 0:
                flag_message = "Consensus flags: {}".format(" ".join(flag_list))

            if self.are_dc_events_expected(uid):
                log_fn = logging.warning
            else:
                log_fn = logging.info

            cname = TallyServer.get_client_display_name(uid)
            cdetail = self.get_client_detail(uid)

            if not self.is_client_control_ok(uid):
                logging.warning("control connection delayed more than {}s for client {} {}"
                                .format(EXPECTED_CONTROL_ESTABLISH_MAX,
                                        cname, c_status))

            if not self.is_last_client_event_recent(uid):
                log_fn("{} for client {} {} {}"
                       .format(format_last_event_time_since(
                                            c_status.get('last_event_time')),
                               cname, c_status, flag_message))

            if time_since_checkin > 3 * self.get_checkin_period() + rtt:
                logging.warning("last checkin was {} for client {} {}"
                                .format(format_elapsed_time_wait(
                                            time_since_checkin, 'at'),
                                        cname, c_status))

            if time_since_checkin > 7 * self.get_checkin_period() + rtt:
                logging.warning("marking dead client {} {}"
                                .format(cname, cdetail))
                c_status['state'] = 'dead'

                if self.collection_phase is not None and self.collection_phase.is_participating(uid):
                    self.collection_phase.lost_client(uid)

                self.clients.pop(uid, None)

    def _get_matching_clients(self, c_type, c_state, c_key=None):
        matching_clients = []
        for uid in self.clients:
            if (self.clients[uid]['type'] == c_type and
                self.clients[uid]['state'] == c_state and
                (c_key is None or c_key in self.clients[uid])):
                matching_clients.append(uid)
        return matching_clients

    def get_idle_dcs(self):
        return self._get_matching_clients('DataCollector', 'idle')

    def get_active_dcs(self):
        return self._get_matching_clients('DataCollector', 'active')

    def get_control_dcs(self):
        '''
        Return the set of DCs that have successfully controlled a tor process.
        This does *not* use is_client_control_ok().
        '''
        return self._get_matching_clients('DataCollector', 'active',
                                          'tor_privcount_version')

    def get_event_dcs(self):
        '''
        Return the set of DCs that have received an event recently.
        See is_last_client_event_recent() for details.
        '''
        matching_clients = []
        control_dcs = self.get_control_dcs()
        for uid in control_dcs:
            if self.is_last_client_event_recent(uid):
                matching_clients.append(uid)
        return matching_clients

    def get_idle_sks(self):
        return self._get_matching_clients('ShareKeeper', 'idle')

    def get_active_sks(self):
        return self._get_matching_clients('ShareKeeper', 'active')

    def count_client_states(self):
        dc_idle = len(self.get_idle_dcs())
        dc_active = len(self.get_active_dcs())
        sk_idle = len(self.get_idle_sks())
        sk_active = len(self.get_active_sks())
        return dc_idle, dc_active, sk_idle, sk_active

    def get_checkin_period(self): # called by protocol
        return self.config['checkin_period']

    def get_status(self): # called by protocol
        dc_idle, dc_active, sk_idle, sk_active = self.count_client_states()

        status = {
            'state' : 'idle' if self.collection_phase is None else 'active',
            'time' : self.idle_time if self.collection_phase is None else self.collection_phase.get_start_ts(),
            'dcs_idle' : dc_idle,
            'dcs_active' : dc_active,
            'dcs_total' : dc_idle+dc_active,
            'dcs_required' : self.config['dc_threshold'],
            'dcs_control' : len(self.get_control_dcs()),
            'dcs_event' : len(self.get_event_dcs()),
            'sks_idle' : sk_idle,
            'sks_active' : sk_active,
            'sks_total' : sk_idle+sk_active,
            'sks_required' : self.config['sk_threshold'],
            'completed_phases' : self.num_completed_collection_phases,
            'continue' : self.config['continue'],
            'delay_until' : self.collection_delay.get_next_round_start_time(
                self.config['noise'],
                self.config['delay_period'],
                max_client_rtt=self.get_max_all_client_rtt(),
                always_delay=self.config['always_delay'],
                tolerance=self.config['sigma_decrease_tolerance']),
            'privcount_version' : get_privcount_version(),
        }

        # we can't know the expected end time until we have started
        if self.collection_phase is not None:
            starting_ts = self.collection_phase.get_start_ts()
            if starting_ts is not None:
                status['expected_end_time'] = starting_ts + self.config['collect_period']

        return status

    def _get_client_item(self, uid, item, status=None, substitute=None):
        '''
        Tries to find item in status, or, if status is None, tries
        self.clients[uid].
        Returns substitute if there is no item.
        '''
        assert uid is not None

        if status is None:
            status = self.clients[uid]

        return status.get(item, substitute)

    @staticmethod
    def get_client_display_name(uid):
        '''
        Returns a display name, based on uid, that is a suitable length for
        logging.
        '''
        # Allow standard-length tor relay nicknames and fingerprints
        # Replace entire hex characters when summarising, not just ...
        return summarise_string(uid, max_len=20, ellipsis='....')

    def get_client_type(self, uid, status=None):
        '''
        Uses _get_client_item to find the client type for uid.
        Returns None if client does not have a type.
        '''
        return self._get_client_item(uid,
                                     'type',
                                     status,
                                     None)

    def get_client_address(self, uid, status=None):
        '''
        Uses _get_client_item to find the remote peer info (hostname and port)
        for uid.
        Returns a placeholder string if client does not have an address.
        '''
        return self._get_client_item(uid,
                                     'client_address',
                                     status,
                                     '(no remote address)')

    def get_client_nickname(self, uid, status=None):
        '''
        Uses _get_client_item to find a fingerprint.
        Returns None if client will never have a nickname, and placeholder
        strings if we know it has no nickname, or we expect a nickname in
        future.
        '''
        if self.get_client_type(uid, status) != 'DataCollector':
            return None

        nick = self._get_client_item(uid,
                                     'nickname',
                                     status,
                                     '(nickname pending)')
        # Distinguish between unknown and known empty nicknames
        if len(nick) == 0:
            nick = '(no nickname)'
        return nick

    def get_client_fingerprint(self, uid, status=None):
        '''
        Uses _get_client_item to find a fingerprint.
        Returns None if client will never have a fingerprint, and a
        placeholder string if we expect a fingerprint in future.
        '''
        if self.get_client_type(uid, status) != 'DataCollector':
            return None

        return self._get_client_item(uid,
                                     'fingerprint',
                                     status,
                                     '(fingerprint pending)')

    def get_client_flag_list(self, uid, status=None):
        '''
        Return the flags for uid in latest status (updated from its latest
        consensus).
        If there are no flags, return an empty list.
        If it's not a Data Collector, return None.
        '''
        if self.get_client_type(uid, status) != 'DataCollector':
            return None

        return self._get_client_item(uid,
                                     'flag_list',
                                     status,
                                     [])

    def get_client_info(self, uid, status=None):
        '''
        Returns a formatted string containing basic information: the
        client's address (if present).
        '''
        return self.get_client_address(uid, status)

    def get_client_detail(self, uid, status=None):
        '''
        Returns a formatted string containing detailed information: the
        client's nickname, address, and fingerprint (if present).
        '''
        if self.get_client_type(uid, status) != 'DataCollector':
            return self.get_client_info(uid, status)

        return "{} {} {}".format(self.get_client_nickname(uid, status),
                                 self.get_client_address(uid, status),
                                 self.get_client_fingerprint(uid, status))

    def get_client_version(self, uid, status=None):
        '''
        Uses _get_client_item to find privcount_version, tor_version, and
        tor_privcount_version.
        Returns a formatted string containing the versions that are present.
        '''
        privcount_version = self._get_client_item(uid,
                                                  'privcount_version',
                                                  status,
                                                  '(no privcount version)')

        # if we're not expecting additional versions, just use privcount
        if self.get_client_type(uid, status) != 'DataCollector':
            return privcount_version

        tor_version = self._get_client_item(uid,
                                            'tor_version',
                                            status,
                                            '(pending)')
        tor_privcount_version = self._get_client_item(uid,
                                                      'tor_privcount_version',
                                                      status,
                                                      '(pending)')

        return ('privcount: {} tor: {} tor privcount: {}'
                .format(privcount_version, tor_version, tor_privcount_version))

    def is_valid_client_version(self, uid, status=None):
        '''
        Check that the version of client is new enough that we want to use it.
        Warn and return False if it is not.
        '''
        cname = TallyServer.get_client_display_name(uid)
        cinfo = self.get_client_info(uid, status)
        cdetail = self.get_client_detail(uid, status)
        cversion = self.get_client_version(uid, status)

        # Reject DC versions 1.0.0 and 1.0.1, they didn't add noise
        client_type = self.get_client_type(uid, status)
        pc_version = self._get_client_item(uid, 'privcount_version',
                                                  status, None)
        pc_version_number, _, _ = pc_version.partition(' ')
        if client_type == 'DataCollector':
            if pc_version_number == '1.0.0' or pc_version_number == '1.0.1':
                logging.warning("Insecure Data Collector PrivCount version {}: {} {}"
                                .format(pc_version_number, cname, cinfo))
                logging.debug("Insecure Data Collector PrivCount version {}: {} detail {} {}"
                              .format(pc_version_number, cname, cdetail,
                                      cversion))
                return False

        return True

    def set_client_status(self, uid, status): # called by protocol
        cname = TallyServer.get_client_display_name(uid)
        cinfo = self.get_client_info(uid, status)
        cdetail = self.get_client_detail(uid, status)
        cversion = self.get_client_version(uid, status)

        # dump the status content at debug level
        logging.debug("{} {} sent status: {}"
                      .format(cname, cdetail, status))
        if uid in self.clients:
            logging.debug("{} {} has stored state: {}"
                          .format(cname, cdetail, self.clients[uid]))

        # Warn and ignore invalid clients
        if not self.is_valid_client_version(uid, status):
            return

        # only data collectors have a fingerprint
        # oldfingerprint is the previous fingerprint for this client (if any)
        # fingerprint is the current fingerprint in the status (if any)
        # If there is no fingerprint, these are None
        oldfingerprint = self.clients.get(uid, {}).get('fingerprint')
        fingerprint = status.get('fingerprint', None)

        # complain if fingerprint changes, and keep the old one
        if (fingerprint is not None and oldfingerprint is not None and
            fingerprint != oldfingerprint):
            logging.warning("Ignoring fingerprint update from {} {} {} (version: {}) state {}: kept old {} ignored new {}"
                            .format(status['type'], cname, cinfo, cversion,
                                    status['state'], oldfingerprint, fingerprint))

        if uid not in self.clients:
            # data collectors don't have nickname, fingerprint, or tor
            # versions until the round starts
            logging.info("new {} {} {} joined and is {}"
                         .format(status['type'], cname, cinfo,
                                 status['state']))

        oldstate = self.clients[uid]['state'] if uid in self.clients else status['state']
        # for each key, replace the client value with the value from status,
        # or, if uid is a new client, initialise uid with status
        self.clients.setdefault(uid, status).update(status)
        # use status['alive'] as the initial value of 'time'
        self.clients[uid].setdefault('time', status['alive'])
        if oldstate != self.clients[uid]['state']:
            self.clients[uid]['time'] = status['alive']
        # always keep the old fingerprint
        if oldfingerprint is not None:
            self.clients[uid]['fingerprint'] = oldfingerprint

        last_event_time = status.get('last_event_time', None)
        last_event_message = ""
        # only log a message if we expect events
        if self.clients[uid]['type'] == 'DataCollector':
            last_event_message = ' ' + format_last_event_time_since(last_event_time)
        logging.info("----client status: {} {} is alive and {} for {}{}"
                     .format(self.clients[uid]['type'], cname,
                             self.clients[uid]['state'],
                             format_elapsed_time_since(self.clients[uid]['time'], 'since'),
                             last_event_message))
        logging.info("----client status: {} detail {} {} version: {}"
                     .format(self.clients[uid]['type'],
                             uid,
                             cdetail,
                             cversion))

    def get_clock_padding(self, client_uids):
        max_delay = max([self.clients[uid]['rtt']+self.clients[uid]['clock_skew'] for uid in client_uids])
        return max_delay + self.get_checkin_period()

    def start_new_collection_phase(self, dc_uids, sk_uids):
        assert self.collection_phase is None

        clock_padding = self.get_clock_padding(dc_uids + sk_uids)

        sk_public_keys = {}
        for uid in sk_uids:
            sk_public_keys[uid] = self.clients[uid]['public_key']

        traffic_model_conf = None
        if 'traffic_model' in self.config:
            traffic_model_conf = self.config['traffic_model']

        # clients don't provide some context until the end of the phase
        # so we'll wait and pass the client context to collection_phase just
        # before stopping it

        # frozensets don't serialise into JSON
        # make a shallow copy of the as_data, so we can modify lists
        as_data = self.config['as_data'].copy()
        as_data['lists'] = [list(c) for c in as_data['lists']]
        self.collection_phase = CollectionPhase(self.config['collect_period'],
                                                self.config['counters'],
                                                traffic_model_conf,
                                                self.config['noise'],
                                                self.config['noise_weight'],
                                                self.config['dc_threshold'],
                                                sk_uids,
                                                sk_public_keys,
                                                dc_uids,
                                                counter_modulus(),
                                                clock_padding,
                                                self.config['max_cell_events_per_circuit'],
                                                self.config['circuit_sample_rate'],
                                                [list(c) for c in self.config['domain_exacts']],
                                                self.config['domain_suffixes'],
                                                [list(c) for c in self.config['country_exacts']],
                                                as_data,
                                                [list(c) for c in self.config['hsdir_store_exacts']],
                                                [list(c) for c in self.config['hsdir_fetch_exacts']],
                                                [list(c) for c in self.config['circuit_failure_exacts']],
                                                [list(c) for c in self.config['onion_address_exacts']],
                                                self.config)
        self.collection_phase.start()

    def stop_collection_phase(self):
        assert self.collection_phase is not None
        self.collection_phase.set_client_status(self.clients)
        self.collection_phase.set_tally_server_status(self.get_status())
        self.collection_phase.stop()
        if self.collection_phase.is_stopped():
            # warn the user if a DC didn't collect any data
            for uid in self.clients:
                if self.clients[uid]['type'] == 'DataCollector':
                    last_event = self.clients[uid].get('last_event_time', None)
                    event_issue = None
                    if last_event is None:
                        event_issue = 'never received any events'
                    elif last_event < self.collection_phase.get_start_ts():
                        event_issue = 'received an event before collection started'
                    if event_issue is not None:
                        cname = TallyServer.get_client_display_name(uid)
                        cdetail = self.get_client_detail(uid)
                        # we could refuse to provide any results here, but they
                        # could still be useful even if they are missing a DC
                        # (or the DC has clock skew). So deliver the results
                        # and allow the operator to decide how to interpret
                        # them
                        logging.warning('Data Collector {} {} {}. Check the results before using them.'
                                        .format(cname, cdetail, event_issue))

            # we want the end time after all clients have definitely stopped
            # and returned their results, not the time the TS told the
            # CollectionPhase to initiate the stop procedure
            # (otherwise, a lost message or down client could delay stopping
            # for a pathological period of time, breaking our assumptions)
            # This also means that the SKs will allow the round to start
            # slightly before the TS allows it, which is a good thing.
            end_time = time()
            self.num_completed_collection_phases += 1
            self.collection_phase.write_results(self.config['results'],
                                                end_time)
            self.collection_delay.set_delay_for_stop(
                not self.collection_phase.is_error(),
                # we can't use config['noise'], because it might have changed
                # since the start of the round
                self.collection_phase.get_noise_config(),
                self.collection_phase.get_start_ts(),
                end_time,
                # if config['delay_period'] has changed, we use it, and warn
                # if it would have made a difference
                self.config['delay_period'],
                max_client_rtt=self.get_max_all_client_rtt(),
                always_delay=self.config['always_delay'],
                tolerance=self.config['sigma_decrease_tolerance'])
            self.collection_phase = None
            self.idle_time = time()

    def get_start_config(self, client_uid):
        '''
        called by protocol
        return None to indicate we shouldnt start the client yet
        '''
        if self.collection_phase is not None:
            return self.collection_phase.get_start_config(client_uid)
        else:
            return None

    def set_start_result(self, client_uid, result_data):
        '''
        called by protocol
        '''
        if self.collection_phase is not None:
            self.collection_phase.store_data(client_uid, result_data,
                                             is_start=True)

    def get_stop_config(self, client_uid):
        '''
        called by protocol
        returns None to indicate we shouldnt stop the client yet
        '''
        if self.collection_phase is not None:
            return self.collection_phase.get_stop_config(client_uid)
        elif client_uid in self.clients and self.clients[client_uid]['state'] == 'active':
            # client is active even though we have no collection phase (could be stale client)
            return {'send_counters' : False}
        else:
            return None

    def set_stop_result(self, client_uid, result_data): # called by protocol
        if self.collection_phase is not None:
            self.collection_phase.store_data(client_uid, result_data,
                                             is_start=False)

class CollectionPhase(object):

    def __init__(self, period, counters_config, traffic_model_config, noise_config,
                 noise_weight_config, dc_threshold_config, sk_uids,
                 sk_public_keys, dc_uids, modulus, clock_padding,
                 max_cell_events_per_circuit, circuit_sample_rate,
                 domain_lists, domain_suffixes, country_lists, as_data,
                 hsdir_store_lists, hsdir_fetch_lists, circuit_failure_lists,
                 onion_address_lists,
                 tally_server_config):
        # the counter bins and configs
        self.counters_config = counters_config
        self.traffic_model_config = traffic_model_config
        self.noise_config = noise_config
        self.noise_weight_config = noise_weight_config
        self.dc_threshold_config = dc_threshold_config

        # the participants
        self.sk_uids = sk_uids
        self.sk_public_keys = sk_public_keys
        self.dc_uids = dc_uids

        # the parameters
        self.period = period
        self.modulus = modulus
        self.clock_padding = clock_padding
        self.max_cell_events_per_circuit = max_cell_events_per_circuit
        self.circuit_sample_rate = circuit_sample_rate

        # the count lists
        self.domain_lists = domain_lists
        self.domain_suffixes = domain_suffixes
        self.country_lists = country_lists
        self.as_data = as_data
        self.hsdir_store_lists = hsdir_store_lists
        self.hsdir_fetch_lists = hsdir_fetch_lists
        self.circuit_failure_lists = circuit_failure_lists
        self.onion_address_lists = onion_address_lists

        # make a deep copy, so we can delete unnecesary keys
        self.tally_server_config = deepcopy(tally_server_config)
        self.tally_server_status = None
        self.client_status = {}
        self.client_config = {}

        # setup some state
        self.state = 'new' # states: new -> starting_dcs -> starting_sks -> started -> stopping -> stopped
        self.starting_ts = None
        self.stopping_ts = None
        self.encrypted_shares = {} # uids of SKs to which we send shares {sk_uid : share_data}
        self.need_shares = set() # uids of DCs from which we still need encrypted shares
        self.final_counts = {} # uids of clients and their final reported counts
        self.need_counts = set() # uids of clients from which we still need final counts
        self.error_flag = False

    def _change_state(self, new_state):
        old_state = self.state
        self.state = new_state
        if old_state != new_state:
            logging.info("collection phase state changed from '{}' to '{}'".format(old_state, new_state))

    def start(self):
        if self.state != "new":
            return

        # we are now starting up
        self.starting_ts = time()

        # we first need to get all encrypted shares from the DCs before we
        # forward them to the SKs
        for uid in self.dc_uids:
            self.need_shares.add(uid)
        self._change_state('starting_dcs')

    def stop(self):
        if self.stopping_ts is None:
            self.stopping_ts = time()

        # main state switch to decide how to stop the phase
        if self.state == 'new':
            self._change_state('stopped')

        elif self.state == 'starting_dcs':
            self.need_shares.clear()
            self.encrypted_shares.clear()

            # need to tell all clients to stop and reset
            self._change_state('stopping')
            for uid in self.dc_uids+self.sk_uids:
                self.need_counts.add(uid)
            self.error_flag = True # when sending STOP, indicate error so we dont get tallies

        elif self.state == 'started' or self.state == 'starting_sks':
            if self.state == 'starting_sks':
                logging.warning("waiting for late responses from some SKs")
            # our noise covers activity independent of the length of the period
            # so we can keep results even if we are ending early
            if self.stopping_ts - self.starting_ts >= self.period:
                logging.info("graceful end to collection phase")
            else:
                logging.warning("premature end to collection phase, results may be less accurate than expected due to the noise that was added (if a client is missing, results may be nonsense)")

            for uid in self.dc_uids+self.sk_uids:
                self.need_counts.add(uid)

            # when sending STOP, indicate that we need tallies
            self.error_flag = False
            self._change_state('stopping')

        elif self.state == 'stopping':
            if len(self.need_counts) == 0:
                self._change_state('stopped')

    def lost_client(self, client_uid):
        '''
        this is called when client_uid isn't responding
        we could mark error_flag as true and abort, or keep counting anyway
        and hope we can recover from the error by adding the local state
        files later... TODO
        '''
        pass

    def store_data(self, client_uid, data, is_start=None):
        '''
        Store the data that client_uid returned in response to a start or
        stop command. If is_start is True, it was a start command.
        '''
        cname = TallyServer.get_client_display_name(client_uid)

        assert is_start is not None
        is_start_string = "start" if is_start else "stop"

        logging.info("recevied {} event from {} while in state {}"
                     .format(is_start_string, cname, self.state))
        logging.debug("recevied {} event from {} while in state {}, data: {}"
                      .format(is_start_string, cname, self.state, data))

        if data is None:
            # this can happen if the SK (or DC) is enforcing a delay because
            # the noise allocation has changed
            logging.info("ignoring missing response in {} event from {} while in state {}"
                         .format(is_start_string, cname, self.state))
            return

        if is_start and client_uid in self.dc_uids:
            # we expect these to be the encrpyted and blinded counts
            # from the DCs that we should forward to the SKs during SK startup
            if self.state != 'starting_dcs':
                logging.warning("processing late response in {} event from {} while in state {}, leaving state unchanged"
                                .format(is_start_string, cname, self.state))

            # don't add a share from the same DC twice
            if client_uid in self.need_shares:
                # collect all shares for each SK together
                shares = data # dict of {sk_uid : share}
                for sk_uid in shares:
                    self.encrypted_shares.setdefault(sk_uid, []).append(shares[sk_uid])
                logging.info("received {} shares from data collector {}"
                             .format(len(shares), cname))

                # mark that we got another one
                self.need_shares.remove(client_uid)
                logging.info("need shares from {} more data collectors".format(len(self.need_shares)))
                if len(self.need_shares) == 0:
                    # ok, we got all of the shares for all SKs, now start the SKs
                    for sk_uid in self.sk_uids:
                        self.need_shares.add(sk_uid)
                    if self.state == 'starting_dcs':
                        self._change_state('starting_sks')

        elif is_start and client_uid in self.sk_uids:
            # we expect confirmation from the SKs that they started successfully
            if self.state != 'starting_sks':
                logging.warning("processing late response in {} event from {} while in state {}, leaving state unchanged"
                                .format(is_start_string, cname, self.state))

            # the sk got our encrypted share successfully
            logging.info("share keeper {} started and received its shares"
                         .format(cname))
            if client_uid in self.need_shares:
                self.need_shares.remove(client_uid)
            if len(self.need_shares) == 0:
                if self.state == 'starting_sks':
                    self._change_state('started')

        elif not is_start:
            # we expect this to be a SK or DC share
            if self.state != 'stopping':
                logging.warning("processing late response in {} event from {} while in state {}"
                                .format(is_start_string, cname, self.state))

            # record the configuration for the client context
            response_config = data.get('Config', None)
            if response_config is not None:
                self.set_client_config(client_uid, response_config)

            if client_uid in self.need_counts:
                # the client got our stop command
                counts = data.get('Counts', None)

                if counts is None:
                    logging.warning("received no counts from {}, final results will not be available"
                                    .format(cname))
                    self.error_flag = True
                elif not self.is_error() and len(counts) == 0:
                    logging.warning("received empty counts from {}, final results will not be available"
                                    .format(cname))
                    self.error_flag = True
                elif not self.is_error():
                    logging.info("received {} counters ({} bins) from stopped client {}"
                                 .format(len(counts), count_bins(counts),
                                         cname))
                    # add up the tallies from the client
                    self.final_counts[client_uid] = counts
                else:
                    logging.warning("received counts: error from stopped client {}"
                                    .format(cname))
                self.need_counts.remove(client_uid)
            else:
                logging.info("ignoring duplicate {} response from client {} while in state {}"
                             .format(is_start_string, cname, self.state))
        else:
            logging.warning("ignoring {} response from client {} while in state {}"
                            .format(is_start_string, cname, self.state))

    def is_participating(self, client_uid):
        return True if client_uid in self.sk_uids or client_uid in self.dc_uids else False

    def is_expired(self):
        if self.starting_ts is None:
            return False
        return True if (time() - self.starting_ts) >= self.period else False

    def is_error(self):
        return self.error_flag

    def is_stopped(self):
        return True if self.state == 'stopped' else False

    def get_noise_config(self):
        return self.noise_config

    def get_start_ts(self):
        return self.starting_ts

    def get_start_config(self, client_uid):
        '''
        Get the starting DC or SK configuration.
        Called by protocol via TallyServer.get_start_config()
        '''
        if not self.is_participating(client_uid) or client_uid not in self.need_shares:
            return None

        cname = TallyServer.get_client_display_name(client_uid)

        if not (self.state == 'starting_dcs' or self.state == 'starting_sks'):
            logging.warning('ignoring protocol request for {} start config in state {}'
                            .format(cname, self.state))
            return None

        config = {}

        if self.state == 'starting_dcs' and client_uid in self.dc_uids:
            # the participants
            config['sharekeepers'] = {}
            for sk_uid in self.sk_public_keys:
                config['sharekeepers'][sk_uid] = b64encode(self.sk_public_keys[sk_uid])

            # the counter configs
            config['counters'] = self.counters_config
            if self.traffic_model_config is not None:
                config['traffic_model'] = self.traffic_model_config
            config['noise'] = self.noise_config
            config['noise_weight'] = self.noise_weight_config

            # the parameters
            config['dc_threshold'] = self.dc_threshold_config
            config['defer_time'] = self.clock_padding
            config['collect_period'] = self.period
            config['max_cell_events_per_circuit'] = self.max_cell_events_per_circuit
            config['circuit_sample_rate'] = self.circuit_sample_rate

            # the count lists
            config['domain_lists'] = self.domain_lists
            config['domain_suffixes'] = self.domain_suffixes
            config['country_lists'] = self.country_lists
            config['as_data'] = self.as_data
            config['hsdir_store_lists'] = self.hsdir_store_lists
            config['hsdir_fetch_lists'] = self.hsdir_fetch_lists
            config['circuit_failure_lists'] = self.circuit_failure_lists
            config['onion_address_lists'] = self.onion_address_lists

            logging.info("sending start comand with {} counters ({} bins) and requesting {} shares to data collector {}"
                         .format(len(config['counters']),
                                 count_bins(config['counters']),
                                 len(config['sharekeepers']),
                                 cname))
            logging.debug("full data collector start config {}".format(config))

        elif self.state == 'starting_sks' and client_uid in self.sk_uids:
            # the participants
            config['shares'] = self.encrypted_shares[client_uid]

            # the counter configs
            config['counters'] = self.counters_config
            if self.traffic_model_config is not None:
                config['traffic_model'] = self.traffic_model_config

            # the parameters
            config['noise'] = self.noise_config
            config['noise_weight'] = self.noise_weight_config
            config['dc_threshold'] = self.dc_threshold_config
            config['collect_period'] = self.period

            logging.info("sending start command with {} counters ({} bins) and {} shares to share keeper {}"
                         .format(len(config['counters']),
                                 count_bins(config['counters']),
                                 len(config['shares']),
                                 cname))
            logging.debug("full share keeper start config {}".format(config))

        return config

    def get_stop_config(self, client_uid):
        if not self.is_participating(client_uid) or client_uid not in self.need_counts:
            return None

        assert self.state == 'stopping'

        cname = TallyServer.get_client_display_name(client_uid)

        config = {'send_counters' : not self.is_error()}
        msg = "without" if self.is_error() else "with"
        logging.info("sending stop command to {} {} request for counters"
                     .format(cname, msg))
        return config

    def set_tally_server_status(self, status):
        '''
        status is a dictionary
        '''
        # make a deep copy, so we can delete unnecesary keys
        self.tally_server_status = deepcopy(status)

    def set_client_status(self, status):
        '''
        status is a dictionary of dictionaries, indexed by UID, and then by the
        attribute: name, fingerprint, ...
        '''
        self.client_status = deepcopy(status)

    def set_client_config(self, uid, config):
        '''
        config is a dictionary, indexed by the attributes: name, fingerprint, ...
        '''
        self.client_config[uid] = deepcopy(config)

    def get_client_types(self):
        '''
        returns a list of unique types of clients in self.client_status
        '''
        types = []
        if self.client_status is None:
            return types
        for uid in self.client_status:
            for k in self.client_status[uid].keys():
                if k == 'type' and not self.client_status[uid]['type'] in types:
                    types.append(self.client_status[uid]['type'])
        return types

    def get_client_context_by_type(self):
        '''
        returns a context for each client by UID, grouped by client type
        '''
        contexts = {}
        # we can't group by type without the type from the status
        if self.client_status is None:
            return contexts
        for type in self.get_client_types():
            for uid in self.client_status:
                if self.client_status[uid].get('type', 'NoType') == type:
                    contexts.setdefault(type, {}).setdefault(uid, {})['Status'] = self.client_status[uid]
                    # remove the (inner) types, because they're redundant now
                    contexts[type][uid]['Status'].pop('type', None)
                    # add the client config as well
                    if self.client_config is not None and uid in self.client_config:
                        contexts[type][uid]['Config'] = self.client_config[uid]
        return contexts

    def get_result_context(self, end_time):
        '''
        the context is written out with the tally results
        '''
        result_context = {}

        # log the times used for the round
        result_time = {}
        # Do we want to round these times?
        # (That is, use begin and end instead?)
        result_time['Start'] = self.starting_ts
        result_time['Stopping'] = self.stopping_ts
        result_time['End'] = end_time
        result_time['CollectStopping'] = self.stopping_ts - self.starting_ts
        result_time['CollectEnd'] = end_time - self.starting_ts
        result_time['StoppingDelay'] = end_time - self.stopping_ts
        # the collect, event, and checkin periods are in the tally server config
        result_time['ClockPadding'] = self.clock_padding
        result_context['Time'] = result_time

        # the bins are listed in each Tally, so we don't duplicate them here
        #result_count_context['CounterBins'] = self.counters_config

        # add the context for the clients that participated in the count
        # this includes all status information by default
        # clients are grouped by type, rather than listing them all by UID at
        # the top level of the context
        if self.client_status is not None:
            result_context.update(self.get_client_context_by_type())

        # now remove any context we are sure we don't want
        for dc in result_context.get('DataCollector', {}).values():
            # We don't need the paths from the configs
            if 'state' in dc.get('Config', {}):
                dc['Config']['state'] = "(state path)"
            if 'secret_handshake' in dc.get('Config', {}):
                dc['Config']['secret_handshake'] = "(secret_handshake path)"
            # or the counters
            if 'counters' in dc.get('Config', {}).get('Start',{}):
                dc['Config']['Start']['counters'] = "(counter bins, no counts)"
            if 'counters' in dc.get('Config', {}).get('Start',{}).get('noise',{}):
                dc['Config']['Start']['noise']['counters'] = "(counter sigmas, no counts)"
            # or the sk public keys
            if 'sharekeepers' in dc.get('Config', {}).get('Start',{}):
                for uid in dc['Config']['Start']['sharekeepers']:
                    dc['Config']['Start']['sharekeepers'][uid] = "(public key)"

        # We don't want the public key in the ShareKeepers' statuses
        for sk in result_context.get('ShareKeeper', {}).values():
            if 'key' in sk.get('Config', {}):
                sk['Config']['key'] = "(key path)"
            if 'state' in sk.get('Config', {}):
                sk['Config']['state'] = "(state path)"
            if 'secret_handshake' in sk.get('Config', {}):
                sk['Config']['secret_handshake'] = "(secret_handshake path)"
            if 'public_key' in sk.get('Status', {}):
                sk['Status']['public_key'] = "(public key)"
            # or the counters
            if 'counters' in sk.get('Config', {}).get('Start',{}):
                sk['Config']['Start']['counters'] = "(counter bins, no counts)"
            if 'counters' in sk.get('Config', {}).get('Start',{}).get('noise',{}):
                sk['Config']['Start']['noise']['counters'] = "(counter sigmas, no counts)"

        # add the status and config for the tally server itself
        result_context['TallyServer'] = {}
        if self.tally_server_status is not None:
            result_context['TallyServer']['Status'] = self.tally_server_status
        # even though the counter limits are hard-coded, include them anyway
        result_context['TallyServer']['Config'] = add_counter_limits_to_config(self.tally_server_config)

        # We don't need the paths from the configs
        if 'cert' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['cert'] = "(cert path)"
        if 'key' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['key'] = "(key path)"
        if 'state' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['state'] = "(state path)"
        if 'secret_handshake' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['secret_handshake'] = \
                "(secret_handshake path)"
        if 'allocation' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['allocation'] = \
                "(allocation path)"
        if 'results' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['results'] = \
                "(results path)"
        # And we don't need the bins, they're duplicated in 'Tally'
        if 'counters' in result_context['TallyServer']['Config']:
            result_context['TallyServer']['Config']['counters'] = "(counter bins, no counts)"
        # but we want the noise, because it's not in Tally

        # we don't want the full list of domains or countries or ASs or ...
        # (the DCs summarise their domains before sending back their
        # start configs, to save bandwidth)
        # The config has already been deepcopied
        PrivCountNode.summarise_config_lists(result_context['TallyServer']['Config'])
        return result_context

    def get_updated_traffic_model(self, tallied_counts):
        '''
        Given the tallied counters in tallied_counts, compute an updated
        TrafficModel config by loading the initial traffic model and updating
        the states of the model based on the traffic model labels that were
        just counted during this round.

        Return None if a traffic model config was not provided as input this round,
        or if there was a problem with the tallied_counts that would prevent us from
        updating the model, or if there was an exception in the traffic model
        update function.

        Return the updated traffic model config on success. The new config can
        be used as input into the next collection round and can be used to
        instantiate another TrafficModel instance.
        '''
        if self.traffic_model_config is None: return None

        # create a TrafficModel object from the original input model config
        tmodel = TrafficModel(self.traffic_model_config)
        all_tmodel_labels = tmodel.get_all_counter_labels()

        # the traffic model class expects counts only, i.e, dict[label] = count
        tmodel_counts = {}
        for label in tallied_counts:
            if label not in all_tmodel_labels:
                logging.warning("tallied counter label {} is not a valid traffic model label"
                                .format(label))
            elif 'bins' not in tallied_counts[label]:
                logging.warning("tallied counters are missing bins for traffic model label {}"
                                .format(label))
            elif len(tallied_counts[label]['bins']) < 1:
                logging.warning("tallied counters have too few bins for traffic model label {}"
                                .format(label))
            elif len(tallied_counts[label]['bins'][0]) < 3:
                logging.warning("tallied counters are missing bin count for traffic model label {}"
                                .format(label))
            else:
                # get the actual count (traffic model only uses 1 bin for each label)
                tmodel_counts[label] = tallied_counts[label]['bins'][0][2]

        # now make sure we got counts for all of the labels
        if len(tmodel_counts) == len(tallied_counts):
            # update the original tmodel based on our new counts, and output. it's
            # OK if this fails, because the counts will be stored in the results
            # context and can be used to update the model after the round ends
            try:
                updated_tmodel_conf = tmodel.update_from_tallies(tmodel_counts)
                return updated_tmodel_conf
            except:
                logging.warning("there was a non-fatal exception in the traffic model update function")
                log_error()
                return None
        else:
            # some problem with counter labels
            logging.warning("the traffic model and tallied counter labels are inconsistent")
            return None

    def write_json_file(self, json_object, path_prefix, filename_prefix, begin, end):
        filepath = os.path.join(path_prefix,
                                "{}.{}-{}.json"
                                .format(filename_prefix, begin, end))
        with open(filepath, 'w') as fout:
            json.dump(json_object, fout, sort_keys=True, indent=4)

        return filepath

    def write_results(self, path_prefix, end_time):
        '''
        Write collections results to a file in path_prefix, including end_time
        in the context.
        '''
        # this should already have been done, but let's make sure
        path_prefix = normalise_path(path_prefix)

        if not self.is_stopped():
            logging.warning("trying to write results before collection phase is stopped")
            return

        # keep going, we want the context for debugging
        tally_was_successful = False
        if len(self.final_counts) <= 0:
            logging.warning("no tally results to write!")
        else:
            tallied_counter = SecureCounters(self.counters_config,
                                             self.modulus,
                                             require_generate_noise=False)
            tally_was_successful = tallied_counter.tally_counters(
                self.final_counts.values())

        begin = int(round(self.starting_ts))
        end = int(round(self.stopping_ts))

        tallied_counts = {}
        # keep going, we want the context for debugging
        if not tally_was_successful:
            logging.warning("problem tallying counters, did all counters and bins match!?")
        else:
            tallied_counts = tallied_counter.detach_counts()

            # For backwards compatibility, write out a "tallies" file
            # This file only has the counts
            self.write_json_file(tallied_counts, path_prefix,
                                 "privcount.tallies", begin, end)

        #logging.info("tally was successful, counts for phase from %d to %d were written to file '%s'", begin, end, filepath)

        # Write out an "outcome" file that adds context to the counts
        # This makes it easier to interpret results later on
        result_info = {}

        if tally_was_successful:
            # add the existing list of counts as its own item
            result_info['Tally'] = tallied_counts

            if self.traffic_model_config is not None:
                # compute the updated traffic model and store in results context
                result_info['UpdatedTrafficModel'] = self.get_updated_traffic_model(tallied_counts)

                # also write out a copy of the new model
                new_model_path = self.write_json_file(result_info['UpdatedTrafficModel'],
                                     path_prefix, "privcount.traffic.model", begin, end)

                # link to the latest version of the traffic model
                # this is useful in the case that we want to keep iterating
                # on the latest model but don't want to manually update the
                # config every round.
                tmplinkname = ".privcount.traffic.model.json.tmp"
                linkname = "traffic.model.latest.json"

                if os.path.exists(tmplinkname):
                    os.remove(tmplinkname)

                os.symlink(new_model_path, tmplinkname)
                os.rename(tmplinkname, linkname)

        # add the context of the outcome as another item
        result_info['Context'] = self.get_result_context(end_time)

        filepath = self.write_json_file(result_info, path_prefix,
                             "privcount.outcome", begin, end)

        logging.info("tally {}, outcome of phase of {} was written to file '{}'"
                     .format(
                     "was successful" if tally_was_successful else "failed",
                     format_interval_time_between(begin, 'from', end),
                     filepath))
        self.final_counts = {}

    def log_status(self):
        message = "collection phase is in '{}' state".format(self.state)

        if self.state == 'starting_dcs':
            message += ", waiting to receive shares from {} DCs: {}".format(len(self.need_shares), ','.join([ TallyServer.get_client_display_name(uid) for uid in self.need_shares]))
        elif self.state == 'starting_sks':
            message += ", waiting to send shares to {} SKs: {}".format(len(self.need_shares), ','.join([ TallyServer.get_client_display_name(uid) for uid in self.need_shares]))
        elif self.state == 'started':
            message += ", running for {}".format(format_elapsed_time_since(self.starting_ts, 'since'))
        elif self.state == 'stopping':
            message += ", trying to stop for {}".format(format_elapsed_time_since(self.stopping_ts, 'since'))
            message += ", waiting to receive counts from {} DCs/SKs: {}".format(len(self.need_counts), ','.join([ TallyServer.get_client_display_name(uid) for uid in self.need_counts]))

        logging.info(message)
