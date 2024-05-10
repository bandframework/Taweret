#!/bin/bash

#
# This script has been adapted from code in the IBCDFO code distribution at
#                  https://github.com/POptUS/IBCDFO
# which is shared under the MIT license with the following copyright and
# permissions:
#
# Interpolation-Based Composite Derivative-Free Optimization (IBCDFO)
# Part of POptUS: Practical Optimization Using Structure
# Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory and UChicago Argonne
# LLC through Argonne National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

#
# This returns exit codes that are compatible with the use of this
# script in CI jobs.
#

SCRIPT_PATH=$(dirname -- "${BASH_SOURCE[0]}")
REPO_PATH=$SCRIPT_PATH/..

declare -a FOLDERS=("tools")

pushd $REPO_PATH &> /dev/null

# Let Python package determine if its code is acceptable
tox -r -e check               || exit $?

# Load virtual env so that flake8 is available and ...
. ./.tox/check/bin/activate   || exit $?

# manually check Python code *not* included in a package
for dir in "${FOLDERS[@]}"; do
    echo
    echo "Check Python code in $dir/* ..."
    pushd $dir &> /dev/null   || exit 1
    flake8 --config=./.flake8 || exit $?
    popd &> /dev/null
done
echo

popd
