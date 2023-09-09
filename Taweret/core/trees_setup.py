"""
Name: trees_setup.py
Author: John Yannotty (yannotty.1@osu.edu)
Desc: Setups up working environment for mixing classes

Start Date: 11/28/22
Version: 1.0
"""

import subprocess
import tempfile
import sys

# -----------------------------------------------
# Trees Configuration
# -----------------------------------------------
# Check to see if .deb is intalled in current working directory


def trees_setup():
    openbt_deb = 'openbt_mixing0.current.deb'

    # Create temp directory to install the file
    wd = tempfile.mkdtemp(prefix="openbtpy_")  # Setting this as a test for now
    wd_openbt_deb = wd + '/' + openbt_deb

    # Set github url (maybe configure with submodule)
    openbt_url = "https://github.com/jcyannotty/OpenBT/raw/main/openbt_mixing0.current_amd64-MPI_Ubuntu_20.04.deb"

    # Get current version name
    openbt_version = openbt_url.split("openbt_mixing")[1]
    openbt_version = openbt_version.split("_amd64")[0] + "-MPI"

    # First determine if the file is available in the required directory
    find_deb_out = subprocess.run(
        ['find', wd, '-name', openbt_deb], capture_output=True)
    find0 = True if not find_deb_out.stdout.decode("utf-8") == "" else False

    # If the package is not found, download it from github ---- clean up with
    # submodule
    if not find0:
        download_out = subprocess.run(
            ["wget", "-O", wd_openbt_deb, openbt_url], capture_output=True)
        if download_out.returncode > 0:
            raise Exception(
                "Cannot download .deb package from openbt github. Please check the openbt_url.")

    # Check the version of the debian package that is installed on this cpu
    version_out = subprocess.run(
        ["dpkg-deb", "-f", wd_openbt_deb, "Version"], capture_output=True)
    version0 = version_out.stdout.decode("utf-8").strip("\n")
    # If the current install is not the current version, then download the new
    # version
    if not version0 == openbt_version:
        download_out = subprocess.run(
            ["wget", "-O", wd_openbt_deb, openbt_url], capture_output=True)
        if download_out.returncode > 0:
            raise Exception(
                "Cannot download .deb package from openbt github. Please check the openbt_url.")
        else:
            new_version = True
    else:
        new_version = False

    # Check if the debian package has been installed
    is_install_out = subprocess.run(
        ["dpkg", "-s", "openbt"], capture_output=True)
    is_install0 = is_install_out.stdout.decode("utf-8").strip("\n")

    # Install the debian package
    if is_install0 == '':
        # The package has never been installed - so install it
        install_out = subprocess.run(
            ["sudo", "apt-get", "install", wd_openbt_deb], capture_output=True)
        if install_out.returncode > 0:
            print("Installation Failed")
        else:
            print("Installation Successful")
    elif new_version:
        # The package has been installed, but its outdated
        # Delete previous install
        print("Deleting previous version of Openbt, please follow command prompts....\n")
        delete_out = subprocess.run(
            ["sudo", "apt-get", "remove", "openbt"], stderr=sys.stderr, stdout=sys.stdout)

        # Install new file
        install_out = subprocess.run(
            ["sudo", "apt-get", "install", wd_openbt_deb], capture_output=True)
        if install_out.returncode > 0:
            print("Installation Failed")
        else:
            print("Installation Successful")
    else:
        print("Current package already installed...")

    # Final Check
    check_install_out = subprocess.run(
        ["dpkg", "-s", "openbt"], capture_output=True)
    check_install0 = check_install_out.stdout.decode("utf-8").strip("\n")

    # NEED TO FIGURE OUT WHAT THE CONDITION IS
    if not check_install0 == "":
        current_version_out = subprocess.run(
            ["dpkg-deb", "-f", wd_openbt_deb, "Version"], capture_output=True)
        current_version = current_version_out.stdout.decode(
            "utf-8").strip("\n")
        print("OpenBT Version: " + current_version + " successfully installed")
    else:
        print("OpenBT failed installation. Please report an issue.")
