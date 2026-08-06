# BAND SDK v0.2 Community Policy Compatibility for Taweret

**Website:** https://github.com/bandframework/Taweret \
**Contact:** as727414@ohio.edu, ingles.27@buckeyemail.osu.edu, liyanage.5@osu.edu, yannotty.1@osu.edu \
**Icon:** https://github.com/bandframework/Taweret/blob/main/logos/taweret_logo.PNG \
**Description:**  Taweret is a python package which implements a variety of Bayesian Model Mixing methodologies. 

**Note:** The Trees module in Taweret uses the openbt Python package, which is also offered as part of the BAND framework, as an external dependence. Please refer to the OpenBT SDK for related information regarding that package.

### Mandatory Policies

**BAND SDK**
| # | Policy                 |Support| Notes                   |
|---|-----------------------|-------|-------------------------|
| 1. | Support BAND community GNU Autoconf, CMake, or other build options. |Full| Taweret is fully written in Python which does not have compatibility with CMake or require GNU Autoconfig. |
| 2. | Have a README file in the top directory that states a specific set of testing procedures for a user to verify the software was installed and run correctly. | Full | None. |
| 3. | Provide a documented, reliable way to contact the development team. |Full| The Taweret team can be contacted via the public issues page Github. |
| 4. | Come with an open-source license |Full| Taweret uses the MIT license.|
| 5. | Provide a runtime API to return the current version number of the software. |Full| Printing `Taweret.__version__` will show the version number.|
| 6. | Provide a BAND team-accessible repository. |Full| https://github.com/bandframework/Taweret |
| 7. | Must allow installing, building, and linking against an outside copy of all imported software that is externally developed and maintained. |Full| None.|
| 8. | Have no hardwired print or IO statements that cannot be turned off. |Full| None.|

### Recommended Policies

| # | Policy                 |Support| Notes                   |
|---|------------------------|-------|-------------------------|
|**R1.**| Have a public repository. |Full| Taweret is a public repository. |
|**R2.**| Free all system resources acquired as soon as they are no longer needed. |Full| None. |
|**R3.**| Provide a mechanism to export ordered list of library dependencies. |None| See the list of dependences in the `pyproject.toml` file. |
|**R4.**| Document versions of packages that it works with or depends upon, preferably in machine-readable form.  |None| See R3. |
|**R5.**| Have SUPPORT, LICENSE, and CHANGELOG files in top directory.  |Partial| The LICENSE is in the top directory, the other two files are not included at this time. |
|**R6.**| Have sufficient documentation to support use and further development.  |Full| Full documentation is provided at https://taweretdocs.readthedocs.io/en/latest/index.html and pedagogical information, including examples, at https://bandframework.github.io/Taweret/. |
|**R7.**| Be buildable using 64-bit pointers; 32-bit is optional. |Full| None. |
|**R8.**| Do not assume a full MPI communicator; allow for user-provided MPI communicator. |N/a| None. |
|**R9.**| Use a limited and well-defined name space (e.g., symbol, macro, library, include). |Full| None.|
|**R10.**| Give best effort at portability to key architectures. |Full| None. |
|**R11.**| Install headers and libraries under `<prefix>/include` and `<prefix>/lib`, respectively. |Full| None.|
|**R12.**| All BAND compatibility changes should be sustainable. |Full| None.|
|**R13.**| Respect system resources and settings made by other previously called packages. |Full| None.|
|**R14.**| Provide a comprehensive test suite for correctness of installation verification. |Partial| Code coverage is now at 68%; improvement will occur in future versions.|
