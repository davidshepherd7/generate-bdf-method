
A script to generate code and symbolic expressions for high order explicit or implicit BDF methods using computational algebra (`sympy`).

The BDF methods are a widely used family of initial value ordinary
differential equation solvers. In normal usage only the implicit methods of
this family are useful, but there are certain cases when the explicit
methods are useful.

The higher order explicit BDF methods are somewhat difficult to calculate
by hand, due to the sheer volume of algebra required, hence the requirement
for this script.



Running
=========

You will need python (tested with python 2.76 but hopefully compatible with
python 3+ as well) and sympy (tested with version 0.7.4.1) installed. On a
Debian-based machine these can be installed with the command

    sudo apt-get install python python-sympy

Then to generate, for example, the 3rd order explicit method simply run

    ./generate-bdf-method.py --order 3 --explicit true

Additional help is available with the `-h` flag.


Testing
=======

Automatic tests are done using `nose`. Install nose with the command

    sudo apt-get install python-nose

Then run all tests with

    nosetests generate-bdf-method.py
