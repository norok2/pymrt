#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyMRT: Useful numerical constants."""
import collections  # Container datatypes

from numpy import pi
from scipy.constants import physical_constants

# ======================================================================
# Proton Gyromagnetic Ratio
_GAMMA = physical_constants['proton gyromag. ratio'][0]  # rad/s/T
_GAMMA_BAR = physical_constants['proton gyromag. ratio over 2 pi'][0]  # Hz/T

# ======================================================================
# Gyromagnetic Ratios of MRI interest
# See:
#   - M A Bernstein; K F King; X J Zhou (2004).
#     Handbook of MRI Pulse Sequences.
#     San Diego: Elsevier Academic Press. p. 960. ISBN 0-12-092861-2.
#   - R C Weast; M J Astle, eds. (1982).
#     Handbook of Chemistry and Physics.
#     Boca Raton: CRC Press. p. E66. ISBN 0-8493-0463-6.
#   - https://en.wikipedia.org/wiki/Gyromagnetic_ratio

GAMMA = collections.OrderedDict((  # rad/s/T
    ('1H', _GAMMA),
    ('2H', 41.065e6),
    ('3He', -203.789e6),
    ('7Li', 103.962e6),
    ('13C', 67.262e6),
    ('14N', 19.331e6),
    ('15N', -27.116e6),
    ('17O', -36.264e6),
    ('19F', 251.662e6),
    ('23Na', 70.761e6),
    ('27Al', 69.763e6),
    ('29Si', -53.190e6),
    ('31P', 108.291e6),
    ('57Fe', 8.681e6),
    ('63Cu', 71.118e6),
    ('67Zn', 16.767e6),
    ('129Xe', -73.997e6),
))

GAMMA_BAR = collections.OrderedDict(  # Hz/T
    (key, 1 * val / (2 * pi)) for key, val in GAMMA.items())

# ======================================================================
# Magnetic Susceptibility Values (SI units)
# See:
#   - Schenck J.F. (1996), "The Role of Magnetic Susceptibility in Magnetic
#     Resonance Imaging: MRI Magnetic Compatibility of the First and Second
#     Kinds", Medical Physics 23 (6):815–850, DOI:10.1118/1.597854.
#   - Arrighini G.P., Maestro M., Moccia R. (1968), "Magnetic Properties of
#     Polyatomic Molecules. I. Magnetic Susceptibility of H2O, NH3, CH4, H2O2",
#     The Journal of Chemical Physics 49 (2):882–889, DOI:10.1063/1.1670155.
#   - Glick R.E. (1961), "ON THE DIAMAGNETIC SUSCEPTIBILITY OF GASES1", The
#     Journal of Physical Chemistry 65 (9):1552–1555, DOI:10.1021/j100905a020.
#   - Wapler M.C., Leupold J., Dragonu I., Elverfeld D. von, Zaitsev M.,
#     Wallrabe U. (2014), "Magnetic Properties of Materials for MR Engineering,
#     Micro-MR and Beyond", Journal of Magnetic Resonance 242 (May):233–242,
#     DOI:10.1016/j.jmr.2014.02.005.
#   - https://en.wikipedia.org/wiki/Magnetic_susceptibility
CHI_V = collections.OrderedDict((  # ppb (SI units)
    ('air@293K,101325Pa', +360),
    ('water@293K,101325Pa', -9035),
))
CHI_V.update(
    collections.OrderedDict((
        ('air', CHI_V['air@293K,101325Pa']),
        ('water', CHI_V['water@293K,101325Pa']),
    )))
