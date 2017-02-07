# intro

PrivCount is an independent implementation of the PrivEx Secret Sharing (S2) variant, that has
been customized in order to be able to aggregate a large variety of statistical counts from Tor
while providing differential privacy guarantees. For more information, see the associated
publication:

```
Safely Measuring Tor
23rd ACM Conference on Computer and Communication Security (CCS 2016)
Rob Jansen and Aaron Johnson
```

For more information about PrivEx, see:

```
PrivEx: Private Collection of Traffic Statistics for Anonymous Communication Networks
21st ACM Conference on Computer and Communications Security (CCS 2014)
Tariq Elahi, George Danezis, and Ian Goldberg
```

# requirements

Base components:

    system libs: libssl libssl-dev cffi
    python libs: pyyaml, twisted, pyopenssl, service-identity, cryptography

    We require OpenSSL version 1.0.2 or later for SHA256-padded RSA encryption.

Optional graphing extensions (required only for the `plot` subcommand):

    system libs: libpng libpng-devel, #TODO this list is incomplete
    python libs: numpy, matplotlib

Optional Tor consensus parsing tool:

    python libs: numpy, stem

System libs can be install with `apt-get`, `yum`, `brew`, etc. Python libs can be installed with `pip`, as we explain below.

# installation

I recommend using virtual environments to isolate the python environment and avoid conflicts.
Run the following from the base directory of this package (i.e., the same location of this README).

    pip install virtualenv
    virtualenv --no-site-packages venv
    source venv/bin/activate
    pip install -r requirements.txt
    # if you want to use the optional privcount plot command
    pip install -r requirements-plot.txt
    # if you want to use the optional compute_fractional_position_weights tool
    pip install -r requirements-weights.txt
    pip install -I .
    test/run_test.sh -I . # run the unit tests to check your setup
    deactivate

Troubleshooting:

If 'pip install virtualenv' fails due to permissions errors, install as root. Using 'sudo -H' before 'pip install' should work.

Some environments (macOS) might need help locating headers and libraries. If so, use 'CFLAGS="-I/opt/local/include" LDFLAGS="-L/opt/local/lib"' (substituting your package manager's path) before pip install.

Some environments (macOS) use the site packages, even if '--no-site-packages' is specified. This can cause failures. Use 'pip install -I' to work around this. 'pip --isolated' might also help, as may 'pip uninstall' outside the virtualenv.

If the encryption unit tests fail with an "UnsupportedAlgorithm" exception, make sure you have cryptography >= 1.4 with OpenSSL >= 1.0.2. You may be using a binary wheel that was compiled with an older OpenSSL version. If so, rebuild and reinstall cryptography using 'pip install -I --no-binary cryptography cryptography'.

# running

To run PrivCount, simply activate the virtual environment that you created earlier and then run PrivCount as normal. For example:

    source venv/bin/activate # enter the virtual environment
    test/test_tor_ctl_event.py <control-port-or-path> # test privcount events
    privcount --help
    ...
    deactivate # exit the virtual environment

# deployment

## PrivCount keys

On first run, PrivCount creates the keys that it needs to run:

The TallyServer creates a RSA key pair for SSL encryption:
    * no configuration is required: clients do not check this key

The TallyServer creates a PrivCount secret handshake key:
    * each ShareKeeper and DataCollector needs to know this key to
      successfully handshake with the TallyServer

Each ShareKeeper creates a RSA key pair for public key encryption:
     * each DataCollector needs to know the SHA256 hash of the public key of
       each ShareKeeper

See doc/PrivCountAuthentication.markdown for more details.

## Tor Control Authentication


PrivCount securely authenticates to tor's control port. This prevents the
control port being used to run commands as the tor user.

Password authentication requires a shared secret configured using the
event_source's control_password option. Cookie authentication requires the
PrivCount user to have read access to tor's cookie file.

See doc/TorControlAuthentication.markdown for more details.

# testing

See `test/README.markdown` for notes about testing PrivCount in a private local deployment, or just run the unit tests:

    source venv/bin/activate # enter the virtual environment
    test/run_test.sh -I .
    deactivate # exit the virtual environment

----------

The notes below are outdated and will no longer work!

# deploying PrivCount entities

Example of the global section for a `privcount-config.yml` file, which all nodes need:

    global:
        start_time: 1452520800 # 2016-01-11 at 2pm UTC
        epoch: 604800 # (1 week = 604800 seconds) the safe time frame of stats collection for all stats
        clock_skew: 300 # seconds - to deal with clock skews and latency

## tally server

Generate key and create self signed cert in a new base directory:

    mkdir privcount_ts
    cd privcount_ts
    touch privcount-config.yml # add above global config in here
    openssl genrsa -out ts.key 4096
    openssl req -new -x509 -key ts.key -out ts.cert -days 1825

Choose an address W1.X1.Y1.Z1 and port P1 that is accessible on the Internet, and append the
following as a new section under the global section of the `privcount-config.yml` file:

    tally_server:
        listen_port: P1 # open port on which to listen for remote connections from TKSes
        key: 'ts.key' # path to the key file
        cert: 'ts.cert' # path to the certificate file
        results: 'results.txt'

Then run PrivCount in tally server mode:

    privcount privcount-config.yml ts

## tally key server

Generate key and create self signed cert in a new base directory:

    mkdir privcount_tks
    cd privcount_tks
    touch privcount-config.yml # add above global config in here
    openssl genrsa -out tks.key 4096
    openssl req -new -x509 -key tks.key -out tks.cert -days 1825

Choose an address W2.X2.Y2.Z2 port P2 that is accessible on the Internet, and append the
following as a new section under the global section of the `privcount-config.yml` file:

    tally_key_server:
        listen_port: P2 # open port on which to listen for remote connections from DCs
        key: 'tks.key' # path to the key file
        cert: 'tks.cert' # path to the certificate file
        tally_server_info: # where the tally server is located
            ip: W1.X1.Y1.Z1
            port: P1

Then run PrivCount in tally key server mode:

    privcount privcount-config.yml tks

## data collector

Create a new base directory:

    mkdir privcount_dc
    cd privcount_dc
    touch privcount-config.yml # add above global config in here

Choose a local port L that will listen for connections from Tor.

data_collector:
    listen_port: L # local port on which to listen for local connections from Tor
    noise_weight: 1.0 # distribute noise among all machines / data collectors
    tally_server_info: # where the tally server is located
        ip: W1.X1.Y1.Z1
        port: P1
    tally_key_server_infos: # where the tally key servers are located
        -
            ip: W2.X2.Y2.Z2
            port: P2
    statistics: ... # see test/privcount-test-config.yaml
