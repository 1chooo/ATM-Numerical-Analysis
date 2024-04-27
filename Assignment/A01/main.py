# -*-coding: utf-8 -*-

"""
Course: AP3021
Assignment: 1
Student id: 109601003
Name: 林群賀
"""

import math

from sympy import Symbol, cos, factorial, pprint

# The ask of the assignment 1:

x = Symbol("x")

k = Symbol("k")
two = Symbol("2")
four = Symbol("4")
six = Symbol("6")
twoK = Symbol("2k")
dot = Symbol("...")

print("The cos(0) tylor polynomial is: \n")
pprint(
    1 
    - x ** 2 / factorial(two) 
    + x ** 4 / factorial(four) 
    - x ** 6 / factorial(six) 
    + dot + ((-1) ** k * x ** (twoK) / factorial(twoK))
)
print()


# Build the factorial mechanism 

def factorial(n: int) -> int:  # Build the factorial mechanism
    result = 1
    for i in range(2, n + 1):
        result *= i

    return result


#Build the error criterion : εs

def count_error_criterion(n) :
    error_criterion = (0.5 * 10 ** (2 - n)) * 100

    return error_criterion


# Build the approximate estimate of the error : εa

def count_approximate_estimate_error(current_approximate, prev_approximate) :
    approximate_error = current_approximate - prev_approximate

    if current_approximate == 0 :    # The first and second value of cosine tylor polynomial is 0.0 and 1.0; therefore we have to pass them.
        return 999
    else :
        approximate_estimate_error = abs((approximate_error / current_approximate) * 100)

    return approximate_estimate_error


# Build the percent relative error : εt

def count_percent_relative_error(true_value, approximation) :
    true_error = true_value - approximation
    percent_relative_error = (true_error / true_value) * 100

    return percent_relative_error


# Build cosine tylor polynomial

def tylorPolynomial(n, x) :
    result = 0

    for i in range(n) : 
        result = result + ((-1) ** i * x ** (2 * i)) / factorial(2 * i)

    return float(result)


# Announce the variable.

PI = math.pi
n = 5
error_criterion = count_error_criterion(n)
loop_count = 0

current_approximate = 0.0
prev_approximate = 0.0
true_value = cos(2 * PI)
print(f"True value of cos(0): {true_value}.")

# print((1.00000000000000 - 0.999978232974615 / 1.00000000000000) * 100)
print(f"The error criterion is: {error_criterion}, and n = {n}.\n\nStart the estimate: \n")


# main

while True :
    prev_approximate = current_approximate
    current_value = tylorPolynomial(loop_count, 2 * PI)
    current_approximate = current_value
    approximate_estimate_error = count_approximate_estimate_error(current_approximate, prev_approximate)
    percent_relative_error = count_percent_relative_error(true_value, current_value)

    if (approximate_estimate_error < error_criterion) :
        loop_count += 1

        print(f"Times: {loop_count}, \ncurrent value: {current_value}, \tapproximate estimate error: {approximate_estimate_error} \tpercent relative error: {percent_relative_error}%.\n")
        print("Here is the error that we accept!!!")
        print(f"Total use {loop_count} times to get the result we want!")

        break
    else :
        loop_count += 1

    print(f"Times: {loop_count}, \ncurrent value: {current_value}, \tapproximate estimate error: {approximate_estimate_error} \tpercent relative error: {percent_relative_error}%.")
