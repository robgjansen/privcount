#!/usr/bin/env python
# See LICENSE for licensing information

import os, json
from privcount.traffic_model import TrafficModel, check_traffic_model_config

# The path to the model file, based on the location of privcount/test
PRIVCOUNT_DIRECTORY = os.environ.get('PRIVCOUNT_DIRECTORY', os.getcwd())
TEST_DIRECTORY = os.path.join(PRIVCOUNT_DIRECTORY, 'test')
MODEL_FILENAME = os.path.join(TEST_DIRECTORY, "traffic.model.json")
TALLIES_FILENAME = os.path.join(TEST_DIRECTORY, "traffic.tallies.json")

'''
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
'''

print "Testing traffic model..."

# now test reading in a model
with open(MODEL_FILENAME, 'r') as inf:
    model = json.load(inf)
with open(TALLIES_FILENAME, 'r') as inf:
    tallied_counts = json.load(inf)

tmod = TrafficModel(model)
assert tmod

print "Here is the list of all counter labels:"
for label in sorted(tmod.get_all_counter_labels()):
    print label

print "Updating traffic model from sample tallies..."

# load in the tallies file and transform the result the same way the tally server does
tmodel_counts ={}
for label in tallied_counts:
    if 'bins' not in tallied_counts[label]:
        print("tallied counters are missing bins for traffic model label {}"
                        .format(label))
    elif len(tallied_counts[label]['bins']) < 1:
        print("tallied counters have too few bins for traffic model label {}"
                        .format(label))
    elif len(tallied_counts[label]['bins'][0]) < 3:
        print("tallied counters are missing bin count for traffic model label {}"
                        .format(label))
    else:
        # get the actual count (traffic model only uses 1 bin for each label)
        tmodel_counts[label] = tallied_counts[label]['bins'][0][2]

updated_model_conf = tmod.update_from_tallies(tmodel_counts)

is_valid = check_traffic_model_config(updated_model_conf)
if is_valid:
    print "Success: Updated model is valid"
else:
    print "Error: Updated model is invalid"
    assert False
