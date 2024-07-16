#!/usr/bin/env python

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

"""
Run the script with -h to obtain more information regarding the script.
"""

import sys
import inspect
import argparse
import importlib

from pathlib import Path

# ----- HARDCODED VALUES
# Exit codes so that this can be used in CI build server
_FAILURE = 1
_SUCCESS = 0


try:
    import Taweret
except ImportError as error:
    print()
    print(f"ERROR: {error.name} Python package not installed")
    print()
    exit(_FAILURE)

def main():
    DEFAULT_VERBOSITY = 0
    VALID_VERBOSITY = [0, 1, 2]

    # ----- SPECIFY COMMAND LINE USAGE
    DESCRIPTION = "Return status of Taweret Python package full testing " \
                  + "as exit code for use with CI\n"
    VERBOSE_HELP = "Verbosity level of pytest logging"
    parser = argparse.ArgumentParser(
                description=DESCRIPTION,
                formatter_class=argparse.RawTextHelpFormatter
             )
    parser.add_argument(
        "--verbose", "-v",
        type=int, choices=VALID_VERBOSITY, default=DEFAULT_VERBOSITY,
        help=VERBOSE_HELP
    )

    # ----- GET COMMAND LINE ARGUMENTS
    args = parser.parse_args()
    verbosity_level = args.verbose

    # ----- PRINT VERSION INFORMATION
    pkg = importlib.metadata.distribution("Taweret")
    location = Path(inspect.getfile(Taweret)).parents[0]

    print()
    print("Name: {}".format(pkg.metadata["Name"]))
    print("Version: {}".format(pkg.metadata["Version"]))
    print("Summary: {}".format(pkg.metadata["Summary"]))
    print("Homepage: {}".format(pkg.metadata["Home-page"]))
    print("License: {}".format(pkg.metadata["License"]))
    print("Python requirements: {}".format(pkg.metadata["Requires-Python"]))
    print("Package dependencies:")
    for dependence in pkg.metadata.get_all("Requires-Dist"):
        print(f"\t{dependence}")
    print("Location: {}".format(location))
    print()
    sys.stdout.flush()

    # ----- RUN FULL TEST SUITE
    return _SUCCESS if Taweret.test(verbosity_level) else _FAILURE


if __name__ == "__main__":
    exit(main())
