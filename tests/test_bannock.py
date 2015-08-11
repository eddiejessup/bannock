#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_bannock
----------------------------------

Tests for `bannock` module.
"""

import unittest

from bannock import model


class TestBannock(unittest.TestCase):

    def test_bannock(self):
        m = model.Model1D(seed=1, dt=0.01, origin_flag=False,
                          rho_0=1.0, v_0=20.0, p_0=1.0,
                          chi=0.5, onesided_flag=True,
                          vicsek_R=1.0, c_D=1000.0, c_sink=0.01, c_source=1.0,
                          L=500.0, dx=50.0)
        m.iterate()
        self.assertTrue(True)
