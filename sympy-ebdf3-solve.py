#! /usr/bin/env python

"""A script to generate code for high order explicit or implicit BDF methods.

By default code for eBDF3 is generated. Play with the code in the main()
function to generate other methods.

Tested with python 2.76 (but hopefully compatible with python 3+) and sympy
version 0.7.4.1.
"""


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import sympy
import sys
import itertools as it
import argparse

from sympy import Rational as sRat
from operator import mul
from functools import partial as par
from functools import reduce


# Define some (global) symbols to use
dts = list(sympy.var('Delta:9', real=True))
dys = list(sympy.var('Dy:9', real=True))
ys = list(sympy.var('y:9', real=True))

# Internally we use the notation 0 = n+1, 1 = n, 2 = n-1, etc. because it makes
# life much easier.

# (with Delta_{n+1} = t_{n+1} - t_n)


def divided_diff(order, ys, dts):
    """Caluclate divided differences of the list of values given
    in ys at points separated by the values in dts.

    Should work with symbols or numbers, only tested with symbols.
    """
    assert(len(ys) == order+1)
    assert(len(dts) == order)

    if order > 1:
        return ((divided_diff(order-1, ys[:-1], dts[:-1])
                 - divided_diff(order-1, ys[1:], dts[1:]))
                /
                sum(dts))
    else:
        return (ys[0] - ys[-1])/(dts[0])


def _steps_diff_to_list_of_dts(a, b):
    """Get t_a - t_b in terms of dts.

    e.g.
    a = 0, b = 2:
    t_0 - t_2 = t_{n+1} - t_{n-1} = dt_{n+1} + dt_n = dts[0] + dts[1]

    a = 2, b = 0:
    t_2 - t_0 = - dts[0] - dts[1]
    """
    # if a and b are in the "wrong" order then it's just the negative of
    # the sum with them in the "right" order.
    if a > b:
        return [-x for x in  _steps_diff_to_list_of_dts(b, a)]

    return dts[a:b]


def _product(l):
    """Return the product (i.e. all entrys multiplied together) of a list or iterator.
    """
    return reduce(mul, l, 1)


def bdf_prefactor(order, derivative_point):
    """Calculate the non-divided difference part of the bdf approximation
    with the derivative known at any integer point. For implicit BDF the
    known derivative is at n+1 (so derivative point = 0), others it is
    further back in time (>0).
    """
    assert(order >= 0)
    assert(derivative_point >= 0)

    if order == 0:
        return 0

    terms = 0

    # For each i in the summation (Note that for most i, for low derivative
    # point the contribution is zero. It's possible to do fancy algebra to
    # speed this up, but it's just not worth it!)
    for i in range(0, order):
        # Get a list of l values for which to calculate the product terms.
        l_list = [l for l in range(0, order) if l != i]

        # Calculate a list of product terms (in terms of Delta t symbols).
        c = [ sum(_steps_diff_to_list_of_dts(derivative_point, b))
              for b in l_list ]

        # Multiply together and add to total
        terms += _product(c)

    return terms


def bdf_method(order, derivative_point):
    """Calculate the bdf approximation for dydt. If implicit approximation
    is at t_{n+1}, otherwise it is at t_n.
    """

    # Each term is just the appropriate order prefactor multiplied by a
    # divided difference. Basically just a python implementation of
    # equation (5.12) on page 400 of Hairer1991.
    def single_term(n):
        return (bdf_prefactor(n, derivative_point)
                * divided_diff(n, ys[:n+1], dts[:n]))

    return sum([single_term(i) for i in range(1, order+1)])


def derive_full_method(order, derivative_point):
    """Return the bdf method's formula for ynp1

    (simplified expression)
    """

    dydt_expr = bdf_method(order, derivative_point)

    # Set equal to dydt at derivative-point-th step, then solve for y_{n+1}
    bdf_method_solutions = sympy.solve(sympy.Eq(dydt_expr,
                                                dys[derivative_point]), y0)

    # Check there's one solution only
    assert(len(bdf_method_solutions) == 1)

    # Convert to nicer form
    our_bdf_method = \
        bdf_method_solutions[0].expand().collect(ys+dys).simplify()

    return our_bdf_method


def main():
    """Construct implicit or explicit bdf methods.

    \nCode notation:
    dtn = size of nth time step
    yn = value of y at nth step
    Dyn = derivative at nth step (i.e. f(t_n, y_n))
    nm1 = n-1, np1 = n+1, etc.
    ** is the power operator
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description=main.__doc__,

    # Don't mess up my formating in the help message
    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--order', action = "store",
                        type=int,
                        help="order of the method to generate",
                        required=True)

    parser.add_argument('--explicit', action = "store",
                        type=bool,
                        help="Generate explicit bdf method? (true/false)",
                        required=True)

    args = parser.parse_args()

    print("I'm computing the",
          "explicit" if args.explicit else "implicit",
          "BDF methd of order", args.order, ".\n\n")


    our_bdf_method = derive_full_method(args.order,
                                        1 if args.explicit else 0)


    print("The symbolic representation is [may require a unicode-enabled terminal]:\n")
    print(sympy.pretty(our_bdf_method))

    print("\n\nThe code is:")
    print(code_gen(our_bdf_method))


def code_gen(sympy_method):
    """Convert sympy formula to code.
    """

    bdf_method_code = str(sympy_method)

    # Replace the sympy variables with variable names consistent with my
    # code in ode.py
    sympy_to_odepy_code_string_replacements = \
        {'Delta0': 'dtn', 'Delta1': 'dtnm1', 'Delta2': 'dtnm2', 'Delta3': 'dtnm3',
         'Dy0': 'dynp1', 'Dy1': 'dyn', 'Dy2': 'dynm1',
         'y0': 'ynp1', 'y1': 'yn', 'y2': 'ynm1', 'y3': 'ynm2', 'y4': 'ynm3'}

    # This is a rubbish way to do mass replace (many passes through the
    # text, any overlapping replaces will cause crazy behaviour) but it's
    # good enough for our purposes.
    for key, val in sympy_to_odepy_code_string_replacements.iteritems():
        bdf_method_code = bdf_method_code.replace(key, val)

    # Check that none of the replacements contain things that will be
    # replaced by other replacement operations. Maybe slow but good to test
    # just in case...
    for _, replacement in sympy_to_odepy_code_string_replacements.iteritems():
        for key, _ in sympy_to_odepy_code_string_replacements.iteritems():
            assert(replacement not in key)

    return "ynp1 = " + bdf_method_code


if __name__ == '__main__':
    sys.exit(main())


# Tests
# ============================================================

# run tests with the command `nosetests [filename]`.

def assert_sym_eq(a, b):
    """Compare symbolic expressions. Note that the simplification algorithm
    is not completely robust: might give false negatives (but never false
    positives).

    Try adding extra simplifications if needed, e.g. add .trigsimplify() to
    the end of my_simp.
    """

    def my_simp(expr):
        # Can't .expand() ints, so catch the zero case separately.
        try:
            return expr.expand().simplify()
        except AttributeError:
            return expr

    print()
    print(sympy.pretty(my_simp(a)))
    print("equals")
    print(sympy.pretty(my_simp(b)))
    print()

    # Try to simplify the difference to zero
    assert (my_simp(a - b) == 0)


def check_const_step(order, exact, derivative_point):

    # Derive bdf method
    b = bdf_method(order, derivative_point)

    # Set all step sizes to be Delta0
    b_const_step = b.subs({k: Delta0 for k in dts})

    # Compare with exact
    assert_sym_eq(exact, b_const_step)


def test_const_step_implicit():
    """Check that the implicit methods are correct for fixed step size by
    comparison with Hairer et. al. 1991 pg 366.
    """

    exacts = [(y0 - y1)/Delta0,
              (sRat(3, 2)*y0 - 2*y1 + sRat(1, 2)*y2)/Delta0,
              (sRat(11, 6)*y0 - 3*y1 + sRat(3, 2)*y2 - sRat(1, 3)*y3)/Delta0,
              (sRat(25, 12)*y0 - 4*y1 + 3*y2 - sRat(4, 3)*y3 + sRat(1, 4)*y4)/Delta0]

    orders = [1, 2, 3, 4]

    for order, exact in zip(orders, exacts):
        yield check_const_step, order, exact, 0


def test_const_step_explicit():

    # Get explicit BDF2 (implicit midpoint)'s dydt approximation G&S pg 715
    a = sympy.solve(-y0 + y1 + (1 + Delta0/Delta1)*Delta0*Dy1
                    - (Delta0/Delta1)**2*(y1 - y2), Dy1)
    assert(len(a) == 1)
    IMR_bdf_form = a[0].subs({k: Delta0 for k in dts})

    orders = [1, 2, 3]
    exacts = [(y0 - y1)/Delta0,
              IMR_bdf_form,
              #Hairer pg 364
              (sRat(1, 3)*y0 + sRat(1, 2)*y1 - y2 + sRat(1, 6)*y3)/Delta0
              ]

    for order, exact in zip(orders, exacts):
        yield check_const_step, order, exact, 1


def test_variable_step_implicit_bdf2():

    # From Gresho and Sani pg 715
    exact = sympy.solve(-(y0 - y1)/Delta0 +
                        (Delta0 / (2*Delta0 + Delta1)) * (y1 - y2)/Delta1 +
                        ((Delta0 + Delta1)/(2*Delta0 + Delta1)) * Dy0, Dy0)

    # Should only be one solution, get it
    assert(len(exact) == 1)
    exact = exact[0]

    # Get the method using my code
    mine = bdf_method(2, 0)

    assert_sym_eq(exact, mine)


def test_variable_step_explicit_bdf2():

    # Also from Gresho and Sani pg 715
    exact = sympy.solve(-y0 + y1 + (1 + Delta0/Delta1)*Delta0*Dy1
                        - (Delta0/Delta1)**2*(y1 - y2), Dy1)

    # Should only be one solution, get it
    assert(len(exact) == 1)
    exact = exact[0]

    # Get the method using my code
    mine = bdf_method(2, 1)

    assert_sym_eq(exact, mine)


def test_list_dts():

    # Check we have a list (not a tuple like before...)
    assert list(
        _steps_diff_to_list_of_dts(2,
                                   0)) == _steps_diff_to_list_of_dts(2,
                                                                     0)
    assert list(
        _steps_diff_to_list_of_dts(0,
                                   2)) == _steps_diff_to_list_of_dts(0,
                                                                     2)

    map(assert_sym_eq, _steps_diff_to_list_of_dts(0, 2), [dts[0], dts[1]])
    map(assert_sym_eq, _steps_diff_to_list_of_dts(2, 0), [-dts[0], -dts[1]])

    assert _steps_diff_to_list_of_dts(2, 2) == []


def test_product():
    assert _product([]) == 1
    assert _product(xrange(1, 11)) == 3628800 # scipy.mpmath.factorial(10)
    assert _product(xrange(0, 101)) == 0
