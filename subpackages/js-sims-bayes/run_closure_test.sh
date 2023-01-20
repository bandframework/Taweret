#!/usr/bin/bash

for i in {0..99}; do
    sed -i "s/fixed_validation_pt=.*/fixed_validation_pt=$i/g" src/configurations.py
    rm -f mcmc/*
    ./src/bayes_mcmc.py 1000 --nwalkers 100 --nburnsteps 500 #|| exit
    ./src/emulator_truth_vs_dob.py #|| exit
done
