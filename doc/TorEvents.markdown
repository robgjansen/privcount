# Tor Events used by PrivCount

PrivCount obtains its Tor usage data via Tor Control Events. These events have
been implemented specifically for PrivCount. PrivCount also uses some other
Controller interfaces.

## PrivCount Overview

### Initialisation

1. PrivCount authenticates to the Control Port (PROTOCOLINFO, AUTHENTICATE),
   using one of the following methods:
   * SAFECOOKIE (AUTHCHALLENGE)
   * HASHEDPASSWORD
   * NULL (if neither other method is available, and tor offers NULL
     authentication)

2. PrivCount discovers some relay information (GETINFO, GETCONF)

### Collection

1. PrivCount waits for the collection round to start
2. PrivCount disables torrc reload on HUP (SETCONF __ReloadTorrcOnSIGHUP=1)
   (This stops the EnablePrivCount option being reset by HUPs.)
3. PrivCount turns on the tor PrivCount code (SETCONF EnablePrivCount=1)
4. PrivCount turns on the tor events required for the collection round
   (SETEVENTS PRIVCOUNT_...)
5. Tor starts sending events as they occur, and PrivCount processes these
   events. Events that started before PrivCount was enabled are ignored.

### Cleanup

1. PrivCount waits for the collection round to stop
2. PrivCount turns off all events (SETEVENTS)
3. PrivCount turns off EnablePrivCount (SETCONF EnablePrivCount=0)
4. PrivCount enables torrc reload on HUP (SETCONF __ReloadTorrcOnSIGHUP=0)
   (This allows operators to reload torrcs between collection rounds.)

## Tor Relay Roles

The information available to a Tor Relay depends on the role that relay plays
in the data transfer. Relays can determine their role in the data transfer
based on the commands they receive.

Role     |  Connection | Circuit | Stream | Cell | Bytes  | Additional
---------|-------------|---------|--------|------|--------|-----------------
Guard~   | Y           | Y       | N/E    | Y    | N/C/E& | Client IP~
Bridge   | Y           | Y       | N/E    | Y    | N/C/E& | Client IP~
Middle   | N/C         | N/C     | N/E    | N/C  | N/C/E& |
Exit     | Y           | Y       | Y      | Y    | Y      | DNS^
DirPort  | N/C         | N/A     | N/A    | N/A  | N/C    | URL*, Client IP$
BEGINDIR | N/C         | N/C     | N/C    | N/C  | N/C    | URL*, Client IP$
HSDir    | N/C         | N/C     | N/C    | N/C  | N/C    | HS Descriptor+
Intro    | Y           | Y       | N/E    | Y    | N/C/E& | Service Keys+
Rend     | Y           | Y       | N/C    | Y    | N/C/E& |
Client@  | N/C         | N/C     | N/C    | N/C  | N/C    | DNS^

Usage:
* Y: Available and Collected
* N/C: Available, but not Collected (We choose not to collect this)
* N/E: Available, but Encrypted (The relay does not have the keys needed to
       decrypt this)
* N/A: Not Applicable (The data does not exist)

Notes:
* \~ "Guard" relays can be connected to clients, or Bridge relays, or other
     relays that aren't in the consensus. Other relays authenticate using RSA
     and ed25519 keys, bridges and clients do not, and can not be
     distinguished.  
* \$ Most Directory requests are performed using a direct connection from the
     client, others are performed using a 3-hop path.  
* \^ Application protocols also leak any unencrypted (meta)data to Exit relays
     and Tor Clients.  
* \* Directory requests contain information about the documents being
     downloaded. Clients request all relay documents, but fetch hidden service
     descriptors as-needed.  
* \+ HSDirs handle hidden service descriptor uploads and downloads over
     BEGINDIR. The service keys can be matched with a hidden service
     descriptor if the onion address is known. Next-generation (v3) hidden
     service descriptors can only be decrypted if the onion address is already
     known. Clients always perform HSDir uploads and downloads over a 3-hop
     connection.  
* \& This role sees encrypted cells, including headers and padding, and can
     not distinguish overheads from content.  
* \@ Onion services connect to the tor network as clients. Relays can also
     perform client activities, like fetching a consensus.   

When relays are relaying cells on a circuit, they can only see that the cell
is a RELAY (or RELAY_EARLY) cell. Everything else is encrypted. (Each relay
maintains a separate mapping that tells it where to forward RELAY cells
received on a particular circuit.)

PrivCount does not currently have a "cell" event. Multiple cells may be
included in each "bytes" event.

## PrivCount Events

PrivCount implements the following events.

PrivCount tries to capture all non-directory Tor network traffic for its
traffic modelling. Some other events are issued selectively based on the type
of traffic: this is a work in progress.

PrivCount ignores events for connections, circuits, and streams that were
created before PrivCount was last enabled. Otherwise, events could have
incomplete cell or byte counts, and we would have a bias towards long-running
events. (PrivCount does not send any events when EnablePrivCount is 0.)

PrivCount measures some overheads, and excludes others.
See doc/TorNetworkOverhead.markdown for details.

### PRIVCOUNT_CIRCUIT_CELL

TODO: expand documentation

        Common:
          EventTimestamp
        Cell and Circuit Common:
          EventTimestamp,
          {Previous,Next}{ChannelId,CircuitId},
          IsOriginFlag, IsEntryFlag, IsMidFlag, IsEndFlag,
          IsExitFlag, IsDirFlag, IsHSDirFlag, IsIntroFlag, IsRendFlag,
          IsHSClientSideFlag, HiddenServiceVersionNumber,
          IsMarkedForCloseFlag
        Cell-Specific:
          IsSentFlag, IsOutboundFlag,
          CellCircuitId, CellCommandString,
          RelayCellPayloadByteCount, RelayCellStreamId, RelayCellCommandString,
          IsRecognizedFlag, WasRelayCryptSuccessfulFlag

There are some differences between the Circuit and Cell position flags:

CREATE and EXTEND cells are always received at the End of the circuit.
But by the time the circuit closes, if the EXTEND is successful, the circuit
is in the Mid position.

When a Client Intro End receives an INTRODUCE cell, if the introduction is
unsuccessful, the client may EXTEND to the next intro point (turning the
failed Client Intro End into a Client Intro Mid).

Limitations:

This event is only sent for circuit cells. Cells at the link layer (VERSIONS,
CERTS, etc.) do not trigger this event.

### PRIVCOUNT_DNS_RESOLVED

This event is sent when an exit receives a client request, including:
* both connect and resolve requests,
* for DNS names, and IPv4 and IPv6 addresses,
* whether the DNS name is in Tor's local DNS cache or not,
* with limited checks on whether the address is a valid DNS name or IP address.  
  (Tor clients and Tor Exits perform some sanity checks before sending an
  address for resolution).

There is *no* filtering of unsuccessful DNS hostname lookups. However,
internal tor network requests (such as directory and onion service requests)
do not use DNS resolution, so they do not trigger this event.

The DNS event is *not* used by any PrivCount counters, but was used as part of
PrivEx's censorship blacklist measurements.

#### Event Format

This event has positional fields: order is important. Fields must not be added
or removed from this event: that would break client code.

It includes the following fields:
* Channel ID
* Circuit ID
* Stream ID
* Requested Host Address (hostname or IPv4 or IPv6)
* Current Timestamp

#### Known Issues

It has the following known issues:
* There is no indication in the event whether the request was successful or
  not, so this event can not be used to count client connections (only client
  requests)  
  https://github.com/privcount/privcount/issues/184
* The resolved address is not included in the event  
  https://github.com/privcount/privcount/issues/184
* These events include relay DirPort self-checks to their own IPv4 addresses  
  https://github.com/privcount/privcount/issues/188

### PRIVCOUNT_STREAM_BYTES_TRANSFERRED

This event is sent when tor reads or writes data from a remote exit stream.

Internal tor network requests (such as onion service requests) and directory
requests do not trigger this event: they are filtered out on the tor side.
Zero reads and writes do not trigger this event. Tor also performs some checks
before reading or writing that may cause this event not to be sent.

Relay DirPort self-checks by remote relays do trigger this event, even though
they are not client traffic. The traffic and connections are
negligible compared with all tor network traffic, but may be significant for
small counters that include IPv4 ports 80 or 9030.

Some tor-side filtering of this event may be necessary for performance
reasons: this event is the most frequent event that tor sends to PrivCount.

The bytes event was originally used to implement the PrivCount traffic model
counters, but it is no longer used for that purpose. (Other counters that
report bytes use the same data as this event, but aggregate it, and send the
totals with the END event.)

#### Event Format

This event has positional fields: order is important. Fields must not be added
or removed from this event: that would break client code.

It includes the following fields:
* Channel ID (if the OR channel has not been cleared)
* Circuit ID (if there is a corresponding OR circuit)
* Stream ID
* Is Outbound Flag
* Number of Bytes
* Current Timestamp

#### Known Issues

It has the following known issues:
* This event includes relay DirPort self-checks to their own IPv4 addresses  
  https://github.com/privcount/privcount/issues/188
* The channel and circuit fields in this event may be missing in some cases  
  https://github.com/privcount/privcount/issues/193
* PrivCount's additional RAM allocations may affect the size of various queues
  and caches. This can lead to dropped cells or a smaller Number of Bytes per
  event
  https://github.com/privcount/privcount/issues/349

### PRIVCOUNT_STREAM_ENDED

This event is sent when tor closes a remote exit stream.

Internal tor network requests (such as onion service requests) and directory
requests do not trigger this event: they are filtered out on the tor side.

Relay DirPort self-checks by remote relays do trigger this event, even though
they are not client traffic. The self-testing traffic and connections are
negligible compared with all tor network traffic, but may be significant for
small counters that include IPv4 ports 80 or 9030.

The end stream event is used by the PrivCount stream, circuit stream, and
circuit activity counters.

#### Event Format

This event has positional fields: order is important.

The "Circuit Exit Stream Number" field was added in PrivCount 2.0.0.

Additional fields must not be added or removed from this event: that would
break client code, and require a major version increment.

It includes the following fields:
* Channel ID (if the OR channel has not been cleared)
* Circuit ID (if there is a corresponding OR circuit)
* Stream ID
* Remote Port
* Total Number of Bytes Read
* Total Number of Bytes Written
* Connection Creation Timestamp
* Current Timestamp
* Remote Host Address (hostname or IPv4 or IPv6)
* Resolved Remote IP Address (IPv4 or IPv6)
* Circuit Exit Stream Number (Starts at 1)

#### Known Issues

It has the following known issues:
* This event includes relay DirPort self-checks to their own IPv4 addresses  
  https://github.com/privcount/privcount/issues/188
* The channel and circuit fields in this event may be missing in some cases  
  https://github.com/privcount/privcount/issues/193
* PrivCount's additional RAM allocations may affect Tor's stream limits
  https://github.com/privcount/privcount/issues/349

### PRIVCOUNT_CIRCUIT_CLOSE

TODO: expand documentation

        Common:
          EventTimestamp
        Cell and Circuit Common:
          {Previous,Next}{ChannelId,CircuitId},
          IsOriginFlag, IsEntryFlag, IsMidFlag, IsEndFlag,
          IsExitFlag, IsDirFlag, IsHSDirFlag, IsIntroFlag, IsRendFlag,
          IsHSClientSideFlag, HiddenServiceVersionNumber,
          IsMarkedForCloseFlag,
          HasReceivedCreateCellFlag, OnionHandshakeType, FailureReasonString,
          ExitStreamCount
        Circuit-Specific:
          CreatedTimestamp, IsLegacyCircuitEndEventFlag,
          StateString, PurposeCode, PurposeString, HSStateString,
          {Previous,Next}Node{IPAddress,Fingerprint,RelayFlagList},
          {IntroClientSink,RendSplice}[All Cell and Circuit Common Fields],
          {Inbound,Outbound}{Sent,Received}CellCount,
          {Inbound,Outbound}Exit{Cell,Byte}Count,
          {Inbound,Outbound}DirByteCount

See test/circuit-failure-reason.txt for FailureReasonStrings and their
meanings.

Limitations:

We can't know anything about the purpose of a circuit if no cells have been
sent on it.

The following purposes are not flagged:
* self-testing (DROP cells)
* unused circuits, including preemptive circuits

There may also be other circuit uses that we don't flag.

We only get exit and directory byte counts on circuits. Other byte counts,
including the various hidden service byte counts, have not yet been
implemented.

To calculate onion service data byte counts, filter events on these flags:
* IsRendFlag=1
* IsHSClientSideFlag=1
  (otherwise byte counts are doubled, because a rend splice consists of two
  circuits, one from the client and one from the service)
Then multiply the OutboundSentCellCount (client to service) and
InboundSentCellCount (service to client)  by 509 (for payload bytes) or
514 (for cell bytes including cell headers). See the tor specification at:
https://gitweb.torproject.org/torspec.git/tree/tor-spec.txt#n64

### PRIVCOUNT_CIRCUIT_ENDED

As of PrivCount 1.1.0, this is a legacy event, use PRIVCOUNT_CIRCUIT_CLOSE in
new code.

This event is sent when tor closes a remote exit circuit.

Internal tor network requests (such as onion service requests) and directory
requests do not trigger this event: they are filtered out on the tor side.
Unused circuits are also filtered out.

Relay DirPort self-checks by remote relays do trigger this event, even though
they are not client traffic. The self-testing traffic and connections are
negligible compared with all tor network traffic, but may be significant for
small counters that include IPv4 ports 80 or 9030.

The circuit ended event is used by the PrivCount circuit, circuit stream, and
circuit activity counters.

#### Event Format

This event has positional fields: order is important. Fields must not be added
or removed from this event: that would break client code.

It includes the following fields:
* Channel ID (if the OR channel has not been cleared)
* Circuit ID
* Total Number of Cells In (Read)
* Total Number of Cells Out (Written)
* Total Number of Exit Bytes Read
* Total Number of Exit Bytes Written
* Circuit Creation Timestamp
* Current Timestamp
* Previous Hop Remote IP Address
* Previous Hop Is Client Flag
* Next Hop Remote IP Address
* Next Hop Is Edge Flag

#### Known Issues

It has the following known issues:
* This event includes relay DirPort self-checks to their own IPv4 addresses  
  https://github.com/privcount/privcount/issues/188
* The channel field in this event may be missing in some cases  
  https://github.com/privcount/privcount/issues/193
* If a Remote IP Address is missing, 0.0.0.0 is used as a placeholder  
  https://github.com/privcount/privcount/issues/196
* PrivCount's additional RAM allocations may affect Tor's circuit limits
  https://github.com/privcount/privcount/issues/349

### PRIVCOUNT_CONNECTION_CLOSE

TODO: expand documentation

Fields:
  Common:
    EventTimestamp
  Lifetime Common:
    CreatedTimestamp
  Connection-Specific:
    ChannelId
    InboundByteCount, OutboundByteCount
    InboundCircuitCount, OutboundCircuitCount
    RemoteIsClientFlag, RemoteIPAddress, RemoteCountryCode,
    RemoteIPAddressConnectionCount
    PeerIPAddress (optional, relay peers only),
    PeerIPAddressConsensusRelayCount

Limitations:

RemoteCountryCode will always be "!!" if there is no configured GeoIP[v6]File.
It can also be "??" for addresses with unknown countries.

PeerIPAddress may not always be the consensus address.

RemoteIPAddress may be different to PeerIPAddress due to NAT,
OutboundBindAddress, and OS routing tables.

PeerIPAddressConsensusRelayCount should be between 0 and 2, but may not always
be in this range for similar reasons.

RemoteIPAddressConnectionCount is not a good measure of client churn, because
it only measures simultaneous connections at a point in time.

### PRIVCOUNT_CONNECTION_ENDED

As of PrivCount 1.2.0, this is a legacy event, use PRIVCOUNT_CONNECTION_CLOSE
in new code.

This event is sent when tor closes a remote OR connection.

Internal tor network requests (such as onion service requests) trigger this
event.

Relay DirPort self-checks by remote relays also trigger this event, even
though they are not client traffic. There are very few self-testing
connections compared with all tor network traffic.

Circuits for these requests are multiplexed over connections from the client
or other relays, so they almost always use existing connections.

Client directory requests do not trigger this event: they are filtered out on
the tor side. (Clients make direct connections for most BEGINDIR requests, so
excluding them makes connection counts more accurate.)

The connection ended event is used by the PrivCount connection and connection
lifetime counters.

#### Event Format

This event has positional fields: order is important. Fields must not be added
or removed from this event: that would break client code.

It includes the following fields:
* Channel ID (if the OR channel has not been cleared)
* Connection Creation Timestamp
* Current Timestamp
* Remote IP Address
* Remote Is Client Flag

Byte counts are not implemented for connections.

#### Known Issues

It has the following known issues:
* This event includes relay DirPort self-checks to their own IPv4 addresses  
  https://github.com/privcount/privcount/issues/188
* The channel field in this event may be missing in some cases  
  https://github.com/privcount/privcount/issues/193
* If a Remote IP Address is missing, 0.0.0.0 is used as a placeholder  
  https://github.com/privcount/privcount/issues/196
* PrivCount's additional RAM allocations may affect Tor's connection limits
  https://github.com/privcount/privcount/issues/349

### PRIVCOUNT_HSDIR_CACHE_STORE

This event is sent when tor attempts to parse and store an onion service
descriptor. Any attempt to upload a service descriptor triggers this event,
even if the descriptor was unparseable or not cached (see CacheReasonString).

Onion Services are also known as Hidden Services. Onion service descriptor
versions 2 and 3 are supported (see HiddenServiceVersionNumber).

Version 3 service descriptors encrypt the onion address. As a subsitute, the
event includes the BlindedEd25519PublicKey identifier, which changes every day,
and has an overlap period.

#### Bias Avoidance

The typical order of events for a single descriptor is:
* PRIVCOUNT_HSDIR_CACHE_STORE CacheReasonString=new ...
* PRIVCOUNT_HSDIR_CACHE_FETCH ... (multiple times)
* PRIVCOUNT_HSDIR_CACHE_STORE CacheReasonString=updated ... (multiple times)
* PRIVCOUNT_HSDIR_CACHE_FETCH ... (multiple times)
* PRIVCOUNT_HSDIR_CACHE_EVICT ...

Different versions of a descriptor can be uploaded multiple times throughout
its assignment to a particular HSDir. Each descriptor series can be identified
on the HSDir by its DescriptorIdBase32String (v2) or BlindedEd25519PublicKey
(v3).

To avoid biasing results towards frequenly updated descriptors, clients should
ignore descriptor events until they have seen CacheReasonString=new for each
DescriptorIdBase32String (v2) or BlindedEd25519PublicKey (v3).

Alternately, to avoid biasing fetch and evict results towards previously
stored descriptors, but accept a bias towards frequently updated descriptors,
clients should ignore descriptor events until they have seen
CacheReasonString=new or CacheReasonString=updated for each
DescriptorIdBase32String (v2) or BlindedEd25519PublicKey (v3).

#### Event Format

This event has tagged fields: order may vary, fields may be optional. Optional
fields in current versions may be removed in future versions.

It includes the following fields:

Mandatory:
* HiddenServiceVersionNumber
* EventTimestamp
* CacheReasonString
* WasAddedToCacheFlag

Optional:
* HasExistingCacheEntryFlag (present if the descriptor was parsed successfully)
* EncodedIntroPointByteCount (present if descriptor was parsed successfully)
* EncodedDescriptorByteCount (may be zero)

Optional in v2, Not Available in v3:
* OnionAddress
* DescriptorIdBase32String (not yet implemented in v3)
* DescriptorCreationTime
* SupportedProtocolBitfield
* RequiresClientAuthFlag
* IntroPointCount (missing if using client auth, or parsing failed)
* IntroPointFingerprintList (missing if using client auth, or parsing failed)

Optional in v3, Not Available in v2:
* BlindedEd25519PublicKeyBase64String
* RevisionNumber
* DescriptorLifetime

#### Known Issues

It has the following known issues:
* PrivCount's additional RAM allocations may affect Tor's HSDir cache
  https://github.com/privcount/privcount/issues/349

### PRIVCOUNT_HSDIR_CACHE_FETCH

TODO: expand documentation

Mandatory:
* HiddenServiceVersionNumber
* EventTimestamp
* CacheReasonString
* CacheQueryByteCount

Optional, Only in Parseable Requests:
* HasCacheEntryFlag

Optional, Only in Cached Descriptors:
* EncodedDescriptorByteCount
* EncodedIntroPointByteCount

Optional in v2, Not Available in v3, Only in Parseable Requests:
* DescriptorIdBase32String (v2 only, not yet implemented in v3)

Optional in v2, Not Available in v3, Only in Cached Descriptors:
* CacheLastServedTime
* OnionAddress (v2 only)
* DescriptorCreationTime
* SupportedProtocolBitfield
* RequiresClientAuthFlag
* IntroPointCount (missing if using client auth)
* IntroPointFingerprintList (missing if using client auth)

Optional in v3, Not Available in v2, Only in Parseable Requests:
* BlindedEd25519PublicKeyBase64String

Optional in v3, Not Available in v2, Only in Cached Descriptors:
* CacheCreatedTime
* RevisionNumber
* DescriptorLifetime

Limitations:

See PRIVCOUNT_HSDIR_CACHE_STORE.

### PRIVCOUNT_VITERBI_PACKETS

This event is emitted once per stream, when the stream closes.
The following fields are sent included in the event:

Mandatory:
* EventTimestamp
* ViterbiPathPackets

The viterbi path is a json-encoded list, where each item in the list
represents a packet 'observation' on a stream. An example encoding
looks something like this:

  ViterbiPathPackets=[["s1";"+";1000];["s0";"+";1000];["s0";"-";1000];["s1";"-";1000];["End";"F";1000]]

  - The first item in the internal list is the state (s0, s1, and End)
  - The second item is the direction code:
    - The '+' means the packet went from the circuit origin and traveled
      exitward.
    - The '-' means the packet went from the destination to the exit and
      traveled clientward.
    - The 'F' means the last packet on the stream.
  - The third item is the delay between the previous packet and this one.

If there was an error in Tor or the stream transferred no exit data,
then the event will be:

  ViterbiPathPackets=[]

### PRIVCOUNT_VITERBI_STREAMS

This event is emitted once per stream, when the stream closes.
The following fields are sent included in the event:

Mandatory:
* EventTimestamp
* ViterbiPathStreams

The viterbi path is a json-encoded list, where each item in the list
represents a stream 'observation' on a circuit. An example encoding
looks something like this:

  # TODO document valid ViterbiPathStreams item

  ViterbiPathStreams=[]

## PrivCount Event Field Format

### Positional Fields

Positional fields are a space-separated list of Value items. Fields MUST be
located using their positions. Value MUST NOT contain spaces, newlines
(C '\r' or '\n'), or NUL characters (C '\0'). Value SHOULD be encoded using
printable characters. New fields MUST NOT be added to existing events: this
breaks older clients.

### Tagged Fields

Tagged fields are a space separated list of Key=Value pairs. Fields MUST be
located using their tag. Field positions MAY vary when other fields are
optional, or when new fields are added in later PrivCount versions.

Key and Value MUST NOT contain spaces, equals signs,
newlines (C '\r' or '\n'), or NUL characters (C '\0'). Key MUST be at least
one character, and SHOULD be a CamelCase ASCII identifier. Value SHOULD be
encoded using printable characters, and MAY be zero-length. The '=' separator
MUST always be present. Fields SHOULD be separated by a single space, and
there SHOULD NOT be any spaces before the event's terminating newline.

New fields SHOULD be added to the end of existing events, so that parsers that
assume event order still work. Old fields SHOULD NOT be removed, so that
parsers that assume event presence still work.

Values that are lists SHOULD use comma as a separator. An empty value is an
ambiguous list: it can mean a list with 0 items (an empty list), or a list
containing a single empty string. Currently, all events and client code
assume it is an empty list.

## PrivCount Event Field Detail

### Channel ID

An unsigned 64-bit unique identifier of a tor channel. This is a persistent,
globally unique identifier for the life of the tor process.

### Circuit ID

An unsigned 32-bit unique identifier of a circuit on a particular OR connection
(channel).

### Stream ID
An unsigned 16-bit unique identifier of a stream on a particular circuit.

### Address
A string that represents the location of a remote site on the Internet. This
may be a hostname, IPv4 address, or IPv6 address.

Some parts of the hidden service protocol also use OnionAddresses.

### Port
An unsigned 16-bit number that represents the port of a remote site on the
Internet.

### Number of Cells
An unsigned 64-bit count of cells in (read) or out (written).

### Number of Bytes
An unsigned 64-bit count of bytes read or written. This byte count uses
saturating arithmetic: values that exceed the maximum are reported as the
maximum value.

### Timestamp
A unix epoch time (UTC) in seconds, to microsecond precision
(6 decimal places). The underlying resolution depends on the operating system.

Creation timestamps will always be after the last time PrivCount was enabled,
to avoid capturing incomplete events.

### Time
A unix epoch time (UTC) in seconds, to integer precision (no decimal places).

### Lifetime
The duration of an event or status in seconds, to integer precision
(no decimal places).

### Is Outbound Flag
A numeric boolean flag: 1 for writes, 0 for reads.

### Is Client Flag
A numeric boolean flag.
True (1) if:
* the remote side used a CREATE_FAST handshake to initiate this connection, or
* the remote side did not perform peer authentication.
False (0) if:
* the remote side used another kind of handshake, or
* the remote side performed peer authentication, or
* the circuit is missing.

### Is Edge Flag
A numeric boolean flag.
True (1) if:
* the edge connection is an exit connection, or
* any stream or pending stream on the circuit is an exit connection, or
* there are no streams and no next channel.
False (0) if:
* the circuit is an origin circuit,
* all streams are non-exit connections,
* the next channel is connected to a relay,
* the circuit and connection are both missing.

### HiddenServiceVersionNumber
An integer giving the hidden service (HS) version. Current versions are:
* HSv2: Legacy hidden services. Supported by all current Tor versions.
* HSv3: Next-generation hidden services. Partially implemented and deployed.

Hidden services are also known as onion services.
This field is named after the HiddenServiceVersion torrc option.

### CacheReasonString

An ASCII string that describes why the descriptor was added or rejected from
the cache:

#### HSDir Store and HSDir Cache

Not added to or fetched from the cache because:
* unencrypted: the request was made on the relay's DirPort (v2 and v3).
* unparseable: the descriptor could not be parsed (v2 and v3). Empty uploads
are considered unparseable.

#### HSDir Store

Added to cache:
* new: the descriptor is new in the cache (v2 and v3).
* updated: the descriptor replaced an existing descriptor in the cache (v2 and v3).

Not added to the cache because:
* badrequest: the request is not to the "publish" URL (v3 only).
* badversion: the request is not version 3 (v3 only).
* duplicate: the encoded descriptor body is identical to the encoded body for
             the currently cached descriptor (v2 only).
* expired: the DescriptorCreationTime is more than 3 days old (v2 only).
* future: the DescriptorCreationTime is more than 1 day into the future
          (v2 only).
* obsolete: the DescriptorCreationTime was less than (v2) or
            RevisionNumber was less than or equal to (v3) the corresponding
            field in the descriptor that is already cached.

#### HSDir Fetch

In cache:
* cached: the descriptor is in the cache (v2 and v3).

Not fetched from the cache because:
* uncached: the descriptor is not in the cache (v2 and v3).

#### Notes

v2 descriptors are cache keyed by DescriptorId, v3 descriptors are cache keyed
by BlindedEd25519PublicKey.

PrivCount's additional RAM allocations may affect cache behaviour.

### HasExistingCacheEntryFlag
A numeric boolean flag.
True (1) if:
* the DescriptorId (v2) or BlindedEd25519PublicKey (v3) was already in the
  cache.
False (0) if:
* the Id or Key was not in the cache.
Unknown (missing) if:
* the descriptor was unparseable.

v2 descriptors are cache keyed by DescriptorId, v3 descriptors are cache keyed
by BlindedEd25519PublicKey.

PrivCount's additional RAM allocations may affect cache behaviour.

### WasAddedToCacheFlag
A numeric boolean flag.
True (1) if:
* the descriptor was added to the cache.
False (0) if:
* the descriptor was not added to the cache, or
* the descriptor was unparseable.

PrivCount's additional RAM allocations may affect cache behaviour.

### OnionAddress
A string containing the service id or onion address (without ".onion").

This is a 16-character base32-encoded string, containing the first 10 bytes of
the SHA1 hash of the service's RSA public key.

The onion address is missing if the descriptor is unparseable, or the RSA
public key is malformed.

### DescriptorIdBase32String
A string containing the base-32 encoded descriptor id, a unique identifier for
the descriptor.

This is *not* the v2 onion address.

In v2, there is one DescriptorId per service address, per replica (x2),
per day for descriptors that use no authentication and basic authentication.

For v2 stealth authentication, each client key has a different DescriptorId,
per service address, per replica, per day.

Not yet implemented in v3.

For details, see:
https://gitweb.torproject.org/torspec.git/tree/rend-spec.txt#n218

### BlindedEd25519PublicKeyBase64String
A string containing the base-64 encoded blinded ed25519 public key, a unique
daily identifier for a service's descriptors (v3 only).

There is one BlindedEd25519PublicKey per service address, per day
(or hsdir-interval consensus parameter). There is a 12-hour overlap period per
day where services upload to two sets of HSDirs: this overlap period starts
when the service downloads the first consensus containin a new SRV. The serice
stops publishing with that SRV ~36 hours later.

For details, see:
https://gitweb.torproject.org/torspec.git/tree/proposals/224-rend-spec-ng.txt#n927

### DescriptorCreationTime
The time that a descriptor was created by the service (v2 only). Onion services
round this value down to the nearest hour.

This is a unix epoch time (UTC) in seconds, to integer precision
(no decimal places).

### RequiresClientAuthFlag
A numeric boolean flag (v2 only).
True (1) if:
* the introduction point section of the descriptor is encrypted (HSDirs can
  not distinguish between encrypted and malformed intro point sections).
False (0) if:
* the intro point section in the descriptor was successfully parsed, and the
  number of intro points was valid, or
* there is no intro point section in the descriptor (if there are no intro
  points, it is not possible to tell if a service requires client
  authentication).
Unknown (missing) if:
* the descriptor was unparseable.

Since the same encryption scheme is used for basic and stealth auth, HSDirs
can not tell the difference between them.

### SupportedProtocolBitfield
A hexadecimal number (prefixed with 0x) representing a bitfield of supported
introduction protocols (v2 only).

This is not the same as HiddenServiceVersionNumber.

All current Tor versions advertise and support:
* Intro version 2: no client authentication
* Intro version 3: adds basic and stealth client authentication
So the bitfield value is always 0x000c (0xc in the event).

Protocol versions 0 and 1 are no longer supported.

This field is currently 16 bits. Tor supports generating descriptors with
versions 0-7, and parsing descriptors with versions 0-15. (See Tor Trac bug
#22534.)

### RevisionNumber
An integer count of the number of descriptor revisions uploaded by the service
(v3 only).

Services increment the revision number each time they upload their descriptor
(at least every 3 hours), regardless of whether the descriptor content has
changed.

### DescriptorLifetime
An integer number of seconds that the descriptor should be cached for
(v3 only).

Services set this to 3 hours (10,800).

### EncodedDescriptorByteCount
An unsigned integer count of bytes in the encoded form of the descriptor.

v2 descriptors are limited to 20kB, and v3 descriptors are limited to 50kB
(or the HSV3MaxDescriptorSize consensus parameter).

The encoding and padding vary between v2 and v3, so byte count comparisons
should be interpreted carefully.

### IntroPointCount
An unsigned integer count of introduction points in the descriptor. Only
available for v2 descriptors with RequiresClientAuthFlag 0. Also missing if the
descriptor or intro point section are unparseable.

v2 descriptors can have between 0 and 10 introduction points.

### IntroPointFingerprintList
A comma-separated list of the introduction point relay fingerprints in the
descriptor. Each fingerprint is the SHA-1 hash of the relay's public RSA key.

If there are zero introduction points, this key is present, and the value is an
empty string. This represents an empty list.

If the IntroPointCount is missing, this key will also be missing.

### EncodedIntroPointByteCount
An unsigned integer count of bytes in the encoded form of the intro point
section of the descriptor.

The intro point encoding varies between v2 and v3, so byte count comparisons
should be interpreted carefully.
