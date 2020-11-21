#!/usr/bin/python

import sys
import unittest

# add individual tests here
from TestAmcPayoffs          import TestAmcPayoffs
from TestCreditModel         import TestCreditModel
from TestHullWhiteModel      import TestHullWhiteModel
from TestHullWhiteMonteCarlo import TestHullWhiteMonteCarlo
from TestPayoffScripting     import TestPayoffScripting
from TestQuasiGaussianModel  import TestQuasiGaussianModel

def test():
    print('Testing HybridMonteCarlo UnitTests:')
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAmcPayoffs))
    suite.addTest(unittest.makeSuite(TestCreditModel))
    suite.addTest(unittest.makeSuite(TestHullWhiteModel))
    suite.addTest(unittest.makeSuite(TestHullWhiteMonteCarlo))
    suite.addTest(unittest.makeSuite(TestPayoffScripting))
    suite.addTest(unittest.makeSuite(TestQuasiGaussianModel))

    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == '__main__':
    test()

