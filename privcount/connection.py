'''
Created on Dec 8, 2016

@author: teor
'''

from collections import Sequence
from exceptions import AttributeError

from twisted.internet import reactor

# Function abstractions for building connections
# Like connectionFromString, but using the reactor model (to avoid a complete
# redesign using endpoints)
# https://twistedmatrix.com/documents/current/core/howto/endpoints.html

# The default listener addresses for each IP version
# These depend on whether we are listening on the loopback interface or not
IP_LISTEN_DEFAULT = {
    4 : { True : '127.0.0.1', False : '0.0.0.0' },
    6 : { True : '::1',       False : '::' }
}

# The default connector addresses for each IP version
# These only make sense if we are connecting to the loopback interface
IP_CONNECT_DEFAULT = {
    4 : '127.0.0.1',
    6 : '::1'
}

def listen(factory, config, ip_local_default=True, ip_version_default=4):
    '''
    Set up factory to listen for connections, based on config, which is a
    dictionary containing listening configuration information.
    IP addresses:
      port: IP port
      ip: IPv4 or IPv6 address (optional)
    UNIX sockets:
      unix: Unix socket path
    If there is a port, but no ip:
    - if ip_local_default is True, IP listeners listen on localhost, otherwise
    - if it is False, they listen on all interfaces, and
    - ip_version_default is the IP version they listen on.
    The default is to listen on IPv4 localhost, to avoid opening unexpected
    public ports.
    Returns a listener object.
    
    config and ip_version_default also accept lists of configs and IP versions.
    (And config accepts an IP and unix config in the same dictionary.)
    This can be hard to debug, because it easily produces duplicate
    connections between the same pair of factories.
    (To avoid these issues, the IP defaults open a single listener on
    localhost.)
    When there are multiple configs, the return value is a list containing
    one listener per listening config. For each config that uses IP version
    defaults, there will be one listener per IP version.
    '''
    # upgrade things to lists if they have been passed as single items
    config = _listify(config)
    ip_version_default = _listify(ip_version_default)
    # now process the lists
    listeners = []
    # the order of the two functions below is irrelevant: all listeners have
    # equal priority regardless of order of creation
    listeners.extend(_listify(listen_unix(factory, config)))
    listeners.extend(_listify(listen_ip(factory, config,
                               ip_local_default, ip_version_default)))
    return _unlistify(listeners)

def listen_unix(factory, config):
    '''
    Internal implementation that opens unix socket listeners for factory based
    on config. See listen() for details.
    '''
    # upgrade config to a list if it has been passed as a single item
    config = _listify(config)
    # now process the list
    listeners = []
    for item in config:
        if not validate_connection_config(item):
            # warn but skip invalid configs
            continue
        if 'unix' in item:
            path = item['unix']
            listeners.append(reactor.listenUNIX(path, factory))
    return _unlistify(listeners)

def listen_ip(factory, config, ip_local_default=True, ip_version_default=4):
    '''
    Internal implementation that opens IP listeners for factory based on
    config. See listen() for details.
    '''
    # upgrade things to lists if they have been passed as single items
    config = _listify(config)
    ip_version_default = _listify(ip_version_default)
    # now process the list
    listeners = []
    for item in config:
        if not validate_connection_config(item):
            # warn but skip invalid configs
            continue
        if 'port' in item:
            port = int(item['port'])
            if 'ip' in item:
                ip = item['ip']
                listeners.append(reactor.listenTCP(port, factory,
                                                   interface=ip))
            else:
                for ip_version in ip_version_default:
                    ip = IP_LISTEN_DEFAULT[ip_version][ip_local_default]
                    listeners.append(reactor.listenTCP(port, factory,
                                                       interface=ip))
    return _unlistify(listeners)

def stopListening(listener):
    '''
    Make listener stop listening.
    If listener is a list, stop all listeners in the list.
    '''
    # upgrade the listener to a list if it has been passed as a single item
    listener = _listify(listener)
    # now process the list
    for item in listener:
        item.stopListening()

def connect(factory, config, ip_local_default=True, ip_version_default=4):
    '''
    Set up factory to connect to service, based on config, which is a
    dictionary containing connection configuration information.
    Known config keys are:
    IP addresses:
      port: IP port
      ip: IPv4 or IPv6 address (optional)
    UNIX sockets:
      unix: Unix socket path
    If there is a port, but no ip:
    - if ip_local_default is True, IP connections connect to localhost on
      IP version ip_version_default.
    - if ip_local_default is False, IP connections must have an ip
      address.
    The default is to try IPv4 localhost.
    Returns a connector object.

    config and ip_version_default also accept lists of configs and IP versions.
    (And config accepts an IP and unix config in the same dictionary.)
    This can be hard to debug, because it easily produces duplicate
    connections between the same pair of factories. In addition, the protocol
    object only stores the last connection opened as the transport, and so all
    events are attributed to that transport, regardless of actual origin.
    (To avoid these issues, the IP defaults open a single connection to
    localhost.)
    When there are multiple connections, the return value is a list of
    connector objects, one per connection config. For each config that uses
    IP version defaults, there will be one connector per IP version.
    '''
    # upgrade things to lists if they have been declared as single items
    config = _listify(config)
    ip_version_default = _listify(ip_version_default)
    # now process the lists
    connectors = []
    # the first connection succeeds and is used by the factory
    # prioritise unix sockets over IP ports, the filesystem is typically more
    # secure
    connectors.extend(_listify(connect_unix(factory, config)))
    connectors.extend(_listify(connect_ip(factory, config,
                                ip_local_default, ip_version_default)))
    return _unlistify(connectors)

def connect_unix(factory, config):
    '''
    Internal implementation that opens unix socket connections for factory
    based on config. See connect() for details.
    '''
    # upgrade config to a list if it has been passed as a single item
    config = _listify(config)
    # now process the list
    connectors = []
    for item in config:
        if not validate_connection_config(item):
            # warn but skip invalid configs
            continue
        if 'unix' in item:
            path = item['unix']
            connectors.append(reactor.connectUNIX(path, factory))
    return _unlistify(connectors)

def connect_ip(factory, config, ip_local_default=True, ip_version_default=4):
    '''
    Internal implementation that opens IP connections for factory based on
    config. See connect() for details.
    '''
    # upgrade things to lists if they have been passed as single items
    config = _listify(config)
    ip_version_default = _listify(ip_version_default)
    # now process the list
    connectors = []
    for item in config:
        if not validate_connection_config(item,
                                       must_have_ip=(not ip_local_default)):
            # warn but skip invalid configs
            continue
        if 'port' in item:
            port = int(item['port'])
            if 'ip' in item:
                ip = item['ip']
                connectors.append(reactor.connectTCP(ip, port, factory))
            elif ip_local_default:
                for ip_version in ip_version_default:
                    ip = IP_CONNECT_DEFAULT[ip_version]
                    connectors.append(reactor.connectTCP(ip, port, factory))
    return _unlistify(connectors)

def disconnect(connector):
    '''
    Disconnect connector.
    If connector is a list, disconnect all connectors in the list.
    '''
    # upgrade the connector to a list if it has been passed as a single item
    connector = _listify(connector)
    # now process the list
    for item in connector:
        item.disconnect()

def validate_connection_config(config, must_have_ip=False):
    '''
    Check that config is valid.
    If must_have_ip is True, config must have an IP address if it has a port.
    Returns False if config is invalid, True otherwise.
    Logs a warning for the first invalid config item found.
    '''
    if config is None:
        logging.warning("Invalid config: None")
        return False
    # upgrade config to a list if it has been passed as a single item
    config = _listify(config)
    # now process the list
    for item in config:
        if item is None:
            logging.warning("Invalid config item: None")
            return False
        if 'port' in item:
            if _config_missing(item, 'port', False):
                logging.warning("Invalid port: missing value")
                return False
            try:
                port = int(item['port'])
            except ValueError as e:
                logging.warning("Invalid port {}: {}".format(port, e))
                return False
            if port <= 0:
                logging.warning("Port {} must be positive".format(port))
                return False
            if must_have_ip and _config_missing(item, 'ip'):
                logging.warning("Port {} must have an IP address".format(port))
                return False
            if 'ip' in item:
                if _config_missing(item, 'ip'):
                    logging.warning("Invalid ip: missing value")
                    return False
                # let the libraries catch other errors later
        elif 'ip' in item:
            logging.warning("IP {} must have a port".format(ip))
            return False
        if 'unix' in item:
            if _config_missing(item, 'unix'):
                logging.warning("Invalid unix path: missing value")
                return False
            # let the libraries catch other errors later
    return True

def choose_a_connection(arg):
    '''
    If arg is a sequence, return an arbitrary item from that sequence.
    Otherwise, return arg.
    '''
    arg = _listify(arg)
    return arg[0]

def _config_missing(config, key, check_len=True):
    '''
    Return True if config is missing key, or if config[key] is None.
    If check_len is True, also return True if len(config[key]) is 0.
    '''
    if key not in config:
        return True
    if config[key] is None:
        return True
    if check_len and len(config[key]) == 0:
        return True
    return False

def _listify(arg):
    '''
    If arg is not a sequence, return it in a one-item list.
    Otherwise, return it unmodified.
    '''
    if not isinstance(arg, (Sequence)):
        return [arg]
    else:
        return arg

def _unlistify(arg):
    '''
    If arg is a sequence with one item, return that item.
    Otherwise, return the list unmodified.
    '''
    if isinstance(arg, (Sequence)) and len(arg) == 1:
        return arg[0]
    else:
        return arg

def transport_info(transport):
    '''
    Return a string describing the remote peer and local endpoint connected
    via transport
    '''
    host_str = transport_host(transport)
    peer_str = transport_peer(transport)
    if host_str is not None and peer_str is not None:
        return "remote: {} local: {}".format(peer_str, host_str)
    elif host_str is not None:
        return host_str
    elif peer_str is not None:
        return peer_str
    else:
        return None

def transport_peer(transport):
    '''
    Return a string describing the remote peer connected to transport
    '''
    peer = transport.getPeer()
    return address_info(peer)

def transport_host(transport):
    '''
    Return a string describing the local endpoint connected to transport
    '''
    try:
        host = transport.getHost()
    except AttributeError:
        return None
    if host is None:
        return None
    return address_info(host)

def address_info(address):
    '''
    Return a string describing the address
    '''
    # Looks like an IPv4Address or IPv6Address
    try:
        # ignore type, it's always TCP
        return "{}:{}".format(address.host, address.port)
    except AttributeError:
        pass
    # Looks like a HostnameAddress
    # (we don't yet support hostnames in connect and listen)
    try:
        return "{}:{}".format(address.hostname, address.port)
    except AttributeError:
        pass
    # Looks like a UNIXAddress
    try:
        # Handle host for UNIXAddress, which is always None
        if address.name is None:
            return None
        else:
            return "{}".format(address.name)
    except AttributeError:
        pass
    if address is None:
        return None
    # Just ask it how it wants to be represented
    return str(address)
