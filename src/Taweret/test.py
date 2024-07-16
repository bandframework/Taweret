import pytest


def test(verbose=0):
    """
    Run the full Taweret test suite.

    Parameters:
    -----------
    :param verbose: Verbosity level with 0 being the least verbose; 2, the
        most.

    Returns:
    --------
    :returns: True if all tests passed; False, otherwise.
    """
    VALID_LEVELS = [0, 1, 2]

    args = []
    if verbose not in VALID_LEVELS:
        raise ValueError(f"verbose must be in {VALID_LEVELS}")
    elif verbose > 0:
        args = ["-" + "v"*verbose]

    args += ["--pyargs", "Taweret.tests"]
    return (pytest.main(args) == 0)
