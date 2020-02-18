#!/bin/bash

echo "Choose a simulation: "
read sim
echo "Define movement function (Move/MoveOld/MoveVelMot): "
read moveCase
echo "Starting Motor Response Calculations"

parallel --bar --eta -j4 "python3 compute-motor-response $sim $moveCase {1} {2} {3}" ::: \
$(seq 0 99) ::: \
$(seq 0 2) ::: \
0 500 2000 7999

