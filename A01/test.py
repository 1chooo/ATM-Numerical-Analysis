import math
from Error import Error
from sympy import *


# Show the Assignment required

x = Symbol("x")
k = Symbol("k")
two = Symbol("2")
four = Symbol("4")
six = Symbol("6")
twoK = Symbol("2k")
dot = Symbol("...")

print("The cosine tylor polynomial is: \n")
pprint(1 - x ** 2 / factorial(two) + x ** 4 / factorial(four) - x ** 6 / factorial(six) + dot + ((-1) ** k * x ** (twoK) / factorial(twoK)))
print()


def tylorPolynomial(n, x) :     # Build cosine tylor polynomial
    sum = 0

    for i in range(n) : 
        sum = sum + ((-1) ** i * x ** (2 * i)) / Error.factorial(2 * i)

    return float(sum)


# Announce the variable.

PI = math.pi
n = 5
errorCriterion = Error.countErrorCriterion(n)
count = 0
currentApproximate = 0.0
previousApproximate = 0.0

print(f"The error criterion is: {errorCriterion}, and n = {n}.\n\nStart the estimate: ")


while True :    # main
    previousApproximate = currentApproximate
    currentValue = tylorPolynomial(count, 2 * PI)
    currentApproximate = currentValue
    approximateEstimateError = Error.countApproximateEstimateError(currentApproximate, previousApproximate)

    if (approximateEstimateError < errorCriterion) :
        count += 1

        print(f"Times: {count}, \tcurrent value: {currentValue}, \tapproximate estimate error: {approximateEstimateError}.\n")
        print(f"Here is the error that we accept!!!")
        print(f"Total use {count} times to get the result we want!")

        break
    else :
        count += 1

    print(f"Times: {count}, \tcurrent value: {currentValue}, \tapproximate estimate error: {approximateEstimateError}.")