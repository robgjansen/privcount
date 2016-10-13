#!/bin/bash

set -e
set -u

# If you have privcount installed in a venv, activate it before running
# this script

# Process arguments
if [ $# -lt 1 -o $# -gt 2 ]; then
  echo "usage: $0 [-I] <privcount-directory>"
  echo "       -I: run 'pip install -I <privcount-directory>' before testing"
  exit 1
elif [ $# -eq 1 ]; then
  PRIVCOUNT_INSTALL=0
  PRIVCOUNT_DIRECTORY="$1"
elif [ $# -eq 2 ]; then
  PRIVCOUNT_INSTALL=1
  PRIVCOUNT_DIRECTORY="$2"
fi

if [ "$PRIVCOUNT_INSTALL" -eq 1 ]; then
  # Install the latest privcount version
  echo "Installing latest version of privcount from '$PRIVCOUNT_DIRECTORY' ..."
  pip install -I "$PRIVCOUNT_DIRECTORY"
fi

cd "$PRIVCOUNT_DIRECTORY/test"

# Run the python-based unit tests
echo "Testing time formats:"
python test_format_time.py
echo ""

echo "Testing keyed hash:"
python test_keyed_random.py
echo ""

echo "Testing counters:"
python test_counters.py
echo ""

echo "Testing traffic model:"
python test_traffic_model.py
echo ""

# Requires a local privcount-patched Tor instance
#python test_tor_ctl_event.py

# Record how long the tests take to run
date
STARTSEC="`date +%s`"

# Move aside the old result files
echo "Moving old results files to '$PRIVCOUNT_DIRECTORY/test/old' ..."
mkdir -p old
mv privcount.* old/ || true

# Then run the injector, ts, sk, and dc
echo "Launching injector, tally server, share keeper, and data collector..."
zcat events.txt.gz | privcount inject --port 20003 --simulate --log - &
privcount ts config.yaml &
privcount sk config.yaml &
privcount dc config.yaml &

# Then wait until they produce a results file
echo "Waiting for results..."
while [ ! -f privcount.outcome.*.json ]; do
  sleep 1
done

# Measure how long the actual tests took
ENDDATE=`date`
ENDSEC="`date +%s`"

# Plot the tallies file
echo "Plotting results..."
privcount plot -d privcount.tallies.*.json test

# And terminate all the privcount processes
echo "Terminating privcount..."
pkill -P $$

# Show how long it took
echo "$ENDDATE"
ELAPSEDSEC=$[ $ENDSEC - $STARTSEC ]
echo "Seconds Elapsed: $ELAPSEDSEC"
