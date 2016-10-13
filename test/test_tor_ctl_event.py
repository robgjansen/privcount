from twisted.internet import reactor
from twisted.internet.protocol import ReconnectingClientFactory
from privcount.protocol import TorControlClientProtocol

TOR_CONTROL_PORT=9050

class TorCtlClient(ReconnectingClientFactory):
    '''
    Connects to a local Tor Control port to test if events of type
    'PRIVCOUNT' are properly exported by Tor.
    '''

    def buildProtocol(self, addr):
        return TorControlClientProtocol(self)
    def handle_event(self, event):
        print "got PRIVCOUNT event: " + event
        return True

reactor.connectTCP("127.0.0.1", TOR_CONTROL_PORT, TorCtlClient())
reactor.run()
