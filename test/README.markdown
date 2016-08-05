# PrivCount - Local Deployment Test

This directory contains the necessary components to run a local deployment of PrivCount and to test
its ability to correctly gather statistics using safe test data.

## Prerequisites

You should have followed the main README to setup and install PrivCount. You should have your
PYTHONPATH setup properly so that you can call `privcount` without providing a full paths to it.

## Running the test

Start the event server that will supply events to the data collector:

    xzcat events.txt.xz | privcount inject --port 20003 --log - --simulate

Start the PrivCount components:

    privcount ts config.yaml
    privcount sk config.yaml
    privcount dc config.yaml

Now wait until the end of the epoch and check the tally results json file published by the
tally server. The results for the 'SanityCheck' counter should indicate a zero count for the
bin representing counts in the range [0, Infinity).

If you have matplotlib installed, you can then visualize the results:

    privcount plot -d <results.json> test

and open the PDF file that was created.
