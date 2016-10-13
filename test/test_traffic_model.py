#!/usr/bin/python

import os, json
from privcount.util import TrafficModel

MODEL_FILENAME="traffic.model.json"

# a sample model
model = {
    'states': ['Blabbing', 'Thinking'],
    'start_probability': {'Blabbing': 0.6, 'Thinking': 0.4},
    'transition_probability': {
        'Blabbing' : {'Blabbing': 0.7, 'Thinking': 0.3},
        'Thinking' : {'Blabbing': 0.4, 'Thinking': 0.6},
    },
    'emission_probability': {
        'Blabbing':{'+': (0.8,0.05), '-': (0.2,0.001)},
        'Thinking':{'+': (0.95,0.0001),'-': (0.05,0.0001)},
    }
}

# write an uncompressed json file
if not os.path.exists(MODEL_FILENAME):
    with open(MODEL_FILENAME, 'w') as outf:
        json.dump(model, outf, sort_keys=True, separators=(',', ': '), indent=2)

del(model)
model = None

print "Testing traffic model..."
print ""

# now test reading in a model
inf = open(MODEL_FILENAME, 'r')
model = json.load(inf)
inf.close()

# the model components
states = model['states']
start_p = model['start_probability']
trans_p = model['transition_probability']
emit_p = model['emission_probability']

tmod = TrafficModel(states, start_p, trans_p, emit_p)

print "Here is the list of all counter labels:"
for label in sorted(tmod.get_counter_labels()):
    print label
print ""

# sample observations
observations = [('+', 20), ('+', 10), ('+', 50), ('+', 1000)]

print "The most likly path through the traffic model given the observations is:"
print "->".join(tmod.run_viterbi(observations))
print ""
