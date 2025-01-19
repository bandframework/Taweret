name: Test package & measure coverage
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  tests:
    #####----- RUN FULL TEST SUITE WITHOUT COVERAGE
    # Test through actual installations for end-to-end testing including
    # packaging.
    strategy:
      matrix:
        os: [ubuntu-20.04, ubuntu-22.04, ubuntu-24.04, macos-13, macos-14, macos-15]
        mpi_impl: ["openmpi", "mpich"]
        python-version: ["3.9", "3.10", "3.11", "3.12", "3.13"]
        exclude:
          # See similar tests for openbtmixing for more info.
          - os: ubuntu-24.04
            mpi_impl: "mpich"
    runs-on: ${{ matrix.os }}
    env:
      CLONE_PATH: ${{ github.workspace }}
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install ${{ matrix.mpi_impl }}
        run: |
          if   [ "${{ runner.os }}" = "Linux" ]; then
             sudo apt-get update
             if   [ "${{ matrix.mpi_impl }}" = "openmpi" ]; then
                sudo apt-get -y install openmpi-bin libopenmpi-dev
             elif [ "${{ matrix.mpi_impl }}" = "mpich" ]; then
                sudo apt-get -y install mpich libmpich-dev
             else
                echo "Cannot install ${{ matrix.mpi_impl }} for Linux"
                exit 1
             fi
          elif [ "${{ runner.os }}" = "macOS" ]; then
             if   [ "${{ matrix.mpi_impl }}" = "openmpi" ]; then
                brew install open-mpi
             elif [ "${{ matrix.mpi_impl }}" = "mpich" ]; then
                brew install mpich
             else
                echo "Cannot install ${{ matrix.mpi_impl }} for macOS"
                exit 1
             fi
          fi
      - name: Setup base Python environment
        run: $CLONE_PATH/.github/workflows/setup_base_python.sh ${{ runner.os }}
      - name: Install Meson build system
        run: $CLONE_PATH/.github/workflows/install_meson.sh ${{ github.workspace }} ${{ runner.os }}
      - name: Run tests without coverage
        run: |
          which mpirun
          which mpicxx
          mpicxx -show
          echo " "
          if   [ "${{ matrix.python-version }}" != "3.9" ]; then
            # TODO: Tests seem to fail with scipy v1.15.0.  Eagerly install with
            # compatible version for now.
            #
            # Python 3.9 only works with scipy < 1.14.0 and will therefore have
            # a compatible scipy version loaded automatically below.
            python -m pip install scipy==1.14.1
          fi
          git clone https://github.com/jcyannotty/OpenBT.git
          pushd OpenBT/openbtmixing_pypkg
          git switch 6_MesonProto
          python -m build --sdist
          python -m pip install -v dist/openbtmixing-1.0.1.tar.gz
          popd
          pushd $CLONE_PATH
          # Since the package is pure Python, the binary wheel should be
          # universal.  Therefore, all pip installs from PyPI should get the
          # wheel rather than the source distribution and we test the wheel.
          python -m build --wheel
          python -m pip install dist/Taweret-*-py3-none-any.whl
          python -c "import Taweret ; print(Taweret.__version__) ; exit(not Taweret.test())"
          popd

#  coverage:
#    #####----- RUN FULL TEST SUITE WITH COVERAGE
#    # Use local editable/developer mode installation so that we are testing such
#    # installations.  For some code coverage services, this can also improve
#    # the information that they present through their web interface.
#    strategy:
#      matrix:
#        os: [ubuntu-latest]
#        python-version: ["3.12"]
#    runs-on: ${{ matrix.os }}
#    env:
#      CLONE_PATH:    ${{ github.workspace }}
#      # These two are used internally by tox
#      COVERAGE_XML:  ${{ github.workspace }}/coverage.xml
#      COVERAGE_HTML: ${{ github.workspace }}/htmlcov
#    steps:
#      - uses: actions/checkout@v4 
#      - name: Setup Python ${{ matrix.python-version }}
#        uses: actions/setup-python@v5
#        with:
#          python-version: ${{ matrix.python-version }}
#      - name: Install OpenMPI
#        run: |
#          sudo apt-get update
#          sudo apt-get -y install openmpi-bin
#      - name: Setup base Python environment
#        run: |
#          $CLONE_PATH/.github/workflows/setup_base_python.sh ${{ matrix.os }}
#      - name: Run tests with coverage
#        run: |
#          pushd $CLONE_PATH
#          tox -r -e coverage,report
#          popd
#      - name: Upload coverage reports to Codecov
#        uses: codecov/codecov-action@v4
#        with:
#          token: ${{ secrets.KEVIN_CODECOV_TOKEN }}
#          slug: bandframework/Taweret
#          file: ${{ env.COVERAGE_XML }}
#          fail_ci_if_error: true
#      - name: Archive code coverage results
#        uses: actions/upload-artifact@v4
#        with:
#          name: code-coverage-results
#          path: |
#             ${{ env.COVERAGE_XML }}
#             ${{ env.COVERAGE_HTML }}
