#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: 32QAM
# Author: Thomas M. Marshall
# GNU Radio version: 3.8.1.0

from gnuradio import analog
from gnuradio import digital
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import osmosdr
import time

# Command line arguments
transmit_gain = float(sys.argv[1])
transmit_frequency = float(sys.argv[2])
transmit_BW = float(sys.argv[3])

class QAM_32(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "32QAM")

        ##################################################
        # Variables
        ##################################################
        self.variable_constellation_0 = variable_constellation_0 = digital.constellation_calcdist([(0.2+0.2j), (-0.2+0.2j), (-0.2-0.2j), (0.2-0.2j), (0.2+0.6j), (-0.2+0.6j), (-0.2-0.6j), (0.2-0.6j), (0.2+1j), (-0.2+1j), (-0.2-1j), (0.2-1j), (0.6+0.2j), (-0.6+0.2j), (-0.6-0.2j), (0.6-0.2j), (0.6+0.6j), (-0.6+0.6j), (-0.6-0.6j), (0.6-0.6j), (0.6+1j), (-0.6+1j), (-0.6-1j), (0.6-1j), (1+0.2j), (-1+0.2j), (-1-0.2j), (1-0.2j), (1+0.6j), (-1+0.6j), (-1-0.6j), (1-0.6j)], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        4, 1).base()
        self.sps = sps = 32
        self.samp_rate = samp_rate = 2.4e6
        self.carrier_freq = carrier_freq = 75e6

        ##################################################
        # Blocks
        ##################################################
        self.osmosdr_sink_0_0 = osmosdr.sink(
            args="numchan=" + str(1) + " " + "hackrf=0"
        )
        self.osmosdr_sink_0_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_sink_0_0.set_sample_rate(samp_rate)
        self.osmosdr_sink_0_0.set_center_freq(transmit_frequency, 0)
        self.osmosdr_sink_0_0.set_freq_corr(0, 0)
        self.osmosdr_sink_0_0.set_gain(13, 0)
        self.osmosdr_sink_0_0.set_if_gain(transmit_gain*8, 0)
        self.osmosdr_sink_0_0.set_bb_gain(0, 0)
        self.osmosdr_sink_0_0.set_antenna('', 0)
        self.osmosdr_sink_0_0.set_bandwidth(0, 0)
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=variable_constellation_0,
            differential=True,
            samples_per_symbol=8,
            pre_diff_code=True,
            excess_bw=transmit_BW,
            verbose=False,
            log=False)
        self.analog_random_uniform_source_x_0 = analog.random_uniform_source_b(0, 255, 0)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_uniform_source_x_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.osmosdr_sink_0_0, 0))

    def get_variable_constellation_0(self):
        return self.variable_constellation_0

    def set_variable_constellation_0(self, variable_constellation_0):
        self.variable_constellation_0 = variable_constellation_0

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.osmosdr_sink_0_0.set_sample_rate(self.samp_rate)

    def get_carrier_freq(self):
        return self.carrier_freq

    def set_carrier_freq(self, carrier_freq):
        self.carrier_freq = carrier_freq



def main(top_block_cls=QAM_32, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.wait()


if __name__ == '__main__':
    main()
