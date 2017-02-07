# PrivCount Authentication

On first run, PrivCount creates the keys that it needs to run:

The TallyServer creates a RSA key pair for SSL encryption:
    * client SSL certificate fingerprint checks are not yet implemented
    * no configuration is required
    * the private key must be kept secret

The TallyServer creates a PrivCount secret handshake key:
    * each ShareKeeper and DataCollector needs to know this key to
      successfully handshake with the TallyServer
    * The configuration item for this key is a file path:
      tally_server:
        secret_handshake: 'keys/secret_handshake.yaml'
      share_keeper:
        secret_handshake: 'keys/secret_handshake.yaml'
      data_collector:
        secret_handshake: 'keys/secret_handshake.yaml'
    * the file should be secretly and authentically transferred to the
      ShareKeeper and DataCollector operators: encrypted, signed emails are a
      good choice
    * the client and server prove knowledge of the key without revealing it,
      using a hash construction similar to tor's SAFECOOKIE authentication 
      (for more details, see privcount/protocol.py)
    * this key must be kept secret: it is equivalent to a symmetric secret key

Each ShareKeeper creates a RSA key pair for public key encryption:
     * each DataCollector needs to know the SHA256 hash of the public key of
       each ShareKeeper
     * the configuration item for this key is an array of hexadecimal
       fingerprints:
       data_collector:
         share_keepers:
         - 'e79ea9173e28e6a54f6d2ec6494c1723a330811652acebbe8d98098ce347d679'
    * the fingerprints can be generated from the ShareKeeper keys using:
        openssl rsa -pubout < keys/sk.pem | sha256sum
    * the fingerprints should be authentically transferred to the
      DataCollector operators: signed emails are a good choice
    * the DataCollectors receive the ShareKeeper public keys via the PrivCount
      protocol, and check that those keys match the configured fingerprints
    * the DataCollectors use these keys to encrypt the blinding shares for
      each ShareKeeper
      (see data_collector.py and share_keeper.py for more details)
    * the public keys and their fingerprints do not need to be kept secret,
      but they must be transferred authentically
    * the private keys on each ShareKeeper must be kept secret
