# BAND SDK v0.4 Community Policy Compatibility for Taweret


> This document summarizes the efforts of current and future BAND member packages to achieve compatibility with the BAND SDK community policies.  Additional details on the BAND SDK are available [here](/resources/sdkpolicies/bandsdk.md) and should be considered when filling out this form. The most recent copy of this template exists [here](/resources/sdkpolicies/template.md).
>
> This file should filled out and placed in the directory in the `bandframework` repository representing the software name appended by `bandsdk`.  For example, if you have a software `foo`, the compatibility file should be named `foobandsdk.md` and placed in the directory housing the software in the `bandframework` repository. No open source code can be included without this file.
>
> All code included in this repository will be open source.  If a piece of code does not contain a open-source LICENSE file as mentioned in the requirements below, then it will be automatically licensed as described in the LICENSE file in the root directory of the bandframework repository.
>
> Please provide information on your compatibility status for each mandatory policy and, if possible, also for recommended policies. If you are not compatible, state what is lacking and what are your plans on how to achieve compliance. For current BAND SDK packages: If you were not fully compatible at some point, please describe the steps you undertook to fulfill the policy. This information will be helpful for future BAND member packages.
>
> To suggest changes to these requirements or obtain more information, please contact [BAND](https://bandframework.github.io/team).
>
> Details on citing the current version of the BAND Framework can be found in the [README](https://github.com/bandframework/bandframework).


**Website:** https://github.com/bandframework/Taweret \
**Contact:** as727414@ohio.edu, ingles.27@buckeyemail.osu.edu, liyanage.5@osu.edu, yannotty.1@osu.edu \
**Icon:** https://github.com/bandframework/Taweret/blob/main/logos/taweret_logo.PNG \
**Description:**  Taweret is a python package which implements a variety of Bayesian Model Mixing methodologies.

### Mandatory Policies

**BAND SDK**
| # | Policy                 |Support| Notes                   |
|---|-----------------------|-------|-------------------------|
| 1. | Support BAND community GNU Autoconf, CMake, or other build options. |Full| The majority of Taweret is written in Python which does not have compatibility with CMake or require GNU Autoconfig. The trees module has an additional step for installing a Ubuntu package and thus does not need compilation. These installation steps are described in the Taweret documentation.|
| 2. | Have a README file in the top directory that states a specific set of testing procedures for a user to verify the software was installed and run correctly. | Full | None. |
| 3. | Provide a documented, reliable way to contact the development team. |Full| The Taweret team can be contacted via the public issues page Github. |
| 4. | Come with an open-source license |Full| Taweret uses the MIT license.|
| 5. | Provide a runtime API to return the current version number of the software. |Full| None.|
| 6. | Provide a BAND team-accessible repository. |Full| https://github.com/bandframework/Taweret |
| 7. | Must allow installing, building, and linking against an outside copy of all imported software that is externally developed and maintained .|Full| None.|
| 8. | Have no hardwired print or IO statements that cannot be turned off. |Full| The trees module prints out one line when the model begins to train. This can be removed if needed.|

### Recommended Policies

| # | Policy                 |Support| Notes                   |
|---|------------------------|-------|-------------------------|
|**R1.**| Have a public repository. |Full| Taweret is a public repository. |
|**R2.**| Free all system resources acquired as soon as they are no longer needed. |Full| None. |
|**R3.**| Provide a mechanism to export ordered list of library dependencies. |None| None. |
|**R4.**| Document versions of packages that it works with or depends upon, preferably in machine-readable form.  |None| None. |
|**R5.**| Have SUPPORT, LICENSE, and CHANGELOG files in top directory.  |Partial| The LICENSE is in the top directory, the other two files are not included at this time. |
|**R6.**| Have sufficient documentation to support use and further development.  |Full| Full documentation is provided at https://bandframework.github.io/Taweret/. |
|**R7.**| Be buildable using 64-bit pointers; 32-bit is optional. |Partial| The trees module depends on a Ubuntu package which is built for 64-bit.|
|**R8.**| Do not assume a full MPI communicator; allow for user-provided MPI communicator. |N/a| None. |
|**R9.**| Use a limited and well-defined name space (e.g., symbol, macro, library, include). |Full| None.|
|**R10.**| Give best effort at portability to key architectures. |Full| Documentation for installing the Ubuntu package required in the trees module is provided. |
|**R11.**| Install headers and libraries under `<prefix>/include` and `<prefix>/lib`, respectively. |Full| None.|
|**R12.**| All BAND compatibility changes should be sustainable. |Full| None.|
|**R13.**| Respect system resources and settings made by other previously called packages. |Full| None.|
|**R14.**| Provide a comprehensive test suite for correctness of installation verification. |Full| None.|
