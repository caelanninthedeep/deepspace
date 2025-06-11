"""
测试包
"""
import unittest
from . import test_environment
from . import test_algorithms
from . import test_utils

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(test_environment.TestMemoryGymEnv))
    test_suite.addTest(unittest.makeSuite(test_algorithms.TestAlgorithms))
    test_suite.addTest(unittest.makeSuite(test_utils.TestUtils))
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite') 