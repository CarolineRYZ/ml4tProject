"""
CS7646 Summer 2021 Project 8 - Test Project
Student Name: Renyu Zhang
GT User ID: rzhang605
GT ID: 903653510
"""


import ManualStrategy as ms
import experiment1 as exp1
import experiment2 as exp2


def author():
    return "CarolineRYZ"


if __name__ == "__main__":
    ms.testManualStrategy(verbose=False)
    exp1.testStrategy(verbose=False)
    exp2.testImpact(verbose=False)


