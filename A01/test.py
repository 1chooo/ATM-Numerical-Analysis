class Error :

    def __init__(self, n) :
        self.n = n

    # Build the factorial mechanism

    def factorial(self) :
        sum = 1

        for i in range(2, self.n + 1) :
            sum *= i

        return sum

    # Build the error criterion : εs

    def errorCriterion() :
        return

    # Build the approximate estimate of the error : εa

    def approximateEstimateError() :
        return

    # Build the percent relative error : εt

    def percentRelativeError() :
        return