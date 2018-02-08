#!/usr/bin/env python
# See LICENSE for licensing information

import os, json
from privcount.traffic_model import TrafficModel

# The path to the model file, based on the location of privcount/test
PRIVCOUNT_DIRECTORY = os.environ.get('PRIVCOUNT_DIRECTORY', os.getcwd())
TEST_DIRECTORY = os.path.join(PRIVCOUNT_DIRECTORY, 'test')
MODEL_FILENAME = os.path.join(TEST_DIRECTORY, "traffic.model.json")

# a sample model
model = {
    'state_space': ['Blabbing', 'Thinking'],
    "observation_space": ["+", "-", "F"],
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

tmod = TrafficModel(model)
assert tmod

print "Here is the list of all counter labels:"
for label in sorted(tmod.get_all_counter_labels()):
    print label
print ""
