#!/bin/bash

echo "randomizing data"
sort --random-sort ../data/output > ../data/random_output
mv ../data/random_output ../data/output
