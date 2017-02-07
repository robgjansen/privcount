# Tor Control Authentication

PrivCount securely authenticates to tor's control port using cookie or password
authentication. This protects against local port accesses (via web pages,
compromised services, or other users) being used to run arbitrary processes as
the tor user.

## Choosing a Control Connection Type

In general, controlling tor over a unix socket is more secure than an IP port:
only processes with the correct filesystem permissions have access to a unix
socket, while any process can connect to a port on localhost.

Unix sockets also perform better than IP sockets, because they have no TCP and
packet overheads.

### Control Unix Socket

Unix sockets are secured by filesystem permissions, but are a bit more complex
to set up. Some OSs configure them automatically.

#### Configure Control Socket in the torrc:
  echo "ControlPort unix:/var/run/tor/control" >> torrc
  (Your OS may have a different default, check the documentation)

#### If you have separate PrivCount and Tor users:
##### Add the PrivCount user to the tor user's group:
  (These user names depend on your distribution and PrivCount config)
  adduser privcount tor
  Add the following flags to the end of the ControlPort line in the torrc:
  GroupWritable

#### If you still get permissions errors:
##### Relax Directory Permissions
  Add the following flags to the end of the ControlPort line in the torrc:
  RelaxDirModeCheck

#### Configure Control Socket in PrivCount
  data_collector:
    event_source:
      unix: '/var/run/tor/control'

### Control Port

Control Ports are vulnerable to cross-protocol attacks, embedded content
attacks (for example, web browsers) and privilege escalation from local
processes.

#### Configure Control Port in the torrc:
  echo "ControlPort 9051" >> torrc

#### Configure Control Port in PrivCount
  data_collector:
    event_source:
      port: 9051

## Configuring Cookie File Authentication

Cookie authentication is secure as long as the cookie file is only readable
by the PrivCount user. The cookie file is owned by the tor user.

This is the simplest and most secure authentication method to configure, but
relies on filesystem and user/group security.

### Configure Cookie Authentication in the torrc:
  echo "CookieAuthentication 1" >> torrc

### If you have separate PrivCount and Tor users:
#### Add the PrivCount user to the tor user's group:
  (These user names depend on your distribution and PrivCount config)
  adduser privcount tor
#### Make the CookieAuthFile group-readable:
  echo "CookieAuthFileGroupReadable 1" >> torrc

## Configuring Password Authentication

Password authentication requires a shared secret configured using the
event_source's control_password option.

### Generate a random 32-byte hexadecimal password using:
  cat /dev/random | hexdump -e '"%x"' -n 32 -v > keys/control_password.txt
### Configure your data collector with the plain text password file:
  data_collector:
    event_source:
      control_password: 'keys/control_password.txt'
  (PrivCount will fail if given an empty or non-existent password file.)
### Hash the password and add it to the torrc
  echo -n "HashedControlPassword " >> torrc
  tor --hash-password `cat keys/control_password.txt` >> torrc

## Implementation Details

Tor can be configured with a cookie file and any number of hashed passwords.
The PrivCount injector and data collector can be configured with a password
file and a cookie file. The injector will accept either method; the data
collector will use cookie authentication if both are configured.

Tor control authentication is not supported when a PrivCount data collector is
configured to connect to multiple event sources. Multiple event sources are
intended for testing purposes only.
