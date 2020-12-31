#!/usr/bin/python

from julia import Main

import sys
import unittest

# add individual tests here
from TestHybridMonteCarlo     import TestHybridModel
from TestHybridQuasiGaussian  import TestHybridQuasiGaussian
from TestSpreadModel          import TestSpreadModel
from TestSwapProduct          import TestSwapProduct

from TestJulia                import TestJulia

runTestQuantLib = True
try:
    from TestQuantLib             import TestQuantLib
except ImportError as e:
   print('Error: Cannot import TestQuantLib. Skip this test.')
   runTestQuantLib = False

def test():
    print('Testing HybridMonteCarlo ComponentTests:')
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestHybridModel))
    suite.addTest(unittest.makeSuite(TestHybridQuasiGaussian))
    suite.addTest(unittest.makeSuite(TestSpreadModel))
    suite.addTest(unittest.makeSuite(TestSwapProduct))

    suite.addTest(unittest.makeSuite(TestJulia))
    if runTestQuantLib:
        suite.addTest(unittest.makeSuite(TestQuantLib))

    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == '__main__':
    test()

