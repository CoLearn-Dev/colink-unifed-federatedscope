import sys

from unifed.frameworks.federatedscope import protocol
from unifed.frameworks.federatedscope.workload_sim import *


def run_protocol():
    print('Running protocol...')
    protocol.pop.run()  # FIXME: require extra testing here

