#!/bin/bash

echo "Choose a simulation: "
read sim
echo "Starting Motor Response Calculations"

parallel --bar --eta -j4 "python3 compute-motor-response $sim {1} {2} {3}" ::: \
$(seq 0 99) ::: \
$(seq 0 2) ::: \
0 3999

