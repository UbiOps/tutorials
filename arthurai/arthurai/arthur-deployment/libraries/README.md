# Libraries

You can use Python packages in your deployment by defining the packages in the `requirements.txt` file as you would
normally do in a local project. The platform will retrieve and install those packages for you during the building phase
using Python pip.

If you want to use custom libraries that are not available on public package repositories, you can install these
libraries into this directory manually. This directory will be added to the system `$PATH`, allowing the deployment to
import them in the usual way.

Your deployment will run in an 64 bit (x86–64) Linux environment. Therefore, if you use any low level libraries that for
example include compiled C extensions or other less portable software, make sure that these are compatible with these
types of systems. Libraries compiled on Windows or Mac machines will often not function properly or encounter
performance issues.
