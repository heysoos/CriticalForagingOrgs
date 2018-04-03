#!/bin/bash

parallel --bar 'python3 compute-motor-response {1}' ::: $(seq 100)

