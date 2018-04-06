#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pymrt.recipes.qmt: quantitative Magnetization Transfer (qMT) computation.

EXPERIMENTAL!
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import itertools  # Functions creating iterators for efficient looping
import warnings  # Warning control
import collections  # Container datatypes

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)

# :: Local Imports
import pymrt as mrt
import pymrt.utils

from pymrt import INFO, DIRS
from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from pymrt import elapsed, report
from pymrt import msg, dbg

from pymrt.recipes import generic
from pymrt.sequences import matrix_algebra
from pymrt.sequences.matrix_algebra import (
    SpinModel, Pulse, Delay, Spoiler, PulseExc, ReadOut,
    MagnetizationPreparation, PulseSequence, SteadyState,
    MultiGradEchoSteadyState, )


# ======================================================================
class MultiMtSteadyState(SteadyState):
    # -----------------------------------
    def __init__(
            self,
            preps,
            *args,
            **kwargs):
        """
        Magnetization Transfer Steady State signals from multiple experiments.

        The different experiments require the same single echo time and
        repetition time, but varying frequency and power level for the
        magnetization preparation (transfer) pulse, as specified by the `preps`
        parameter.

        The pulse sequence of the repeating block for the steady state
        sequence must be specified, as detailed in
        `pymrt.sequences.bloch_sim.SteadyState`.

        Args:
            preps ():
            *args ():
            **kwargs ():
        """
        SteadyState.__init__(self, *args, **kwargs)
        idx = self.get_unique_pulses(('MagnetizationPreparation',))
        if hasattr(self, 'idx'):
            self._idx.update(idx)
        self.preps = preps

    # -----------------------------------
    @staticmethod
    def get_prep_labels():
        return 'PrepFrequency', 'PrepFlipAngle'

    # -----------------------------------
    @staticmethod
    def get_prep_units():
        return 'Hz', 'deg'

    # -----------------------------------
    def signal(
            self,
            spin_model,
            *args,
            **kwargs):
        base_p_ops = self.propagators(spin_model, *args, **kwargs)
        te_p_ops = [
            Delay(t_d).propagator(spin_model, *args, **kwargs)
            for t_d in self._pre_post_delays(self.te, self._t_ro, self._t_pexc)]
        mt_pulse = self.pulses[self._idx['MagnetizationPreparation']]
        f_c = mt_pulse.carrier_freq
        mt_p_ops = [
            mt_pulse.set_carrier_freq(f_c + df).set_flip_angle(fa).propagator(
                spin_model, *args, **kwargs)
            for df, fa in self.preps]
        central_p_ops_list = [
            self._p_ops_reorder(
                self._p_ops_subst(
                    base_p_ops, self._idx['MagnetizationPreparation'],
                    mt_p_op),
                self._idx['ReadOut'])
            for mt_p_op in mt_p_ops]
        s_arr = np.array([
            self._signal(
                spin_model,
                self._propagator(
                    [te_p_ops[0]] + central_p_ops + [te_p_ops[1]],
                    self.n_r))
            for central_p_ops in central_p_ops_list])
        return s_arr


# ======================================================================
class MultiMtSteadyState2(SteadyState):
    # -----------------------------------
    def __init__(
            self,
            freqs,
            fas,
            *args,
            **kwargs):
        """

        Args:
            kernel (PulseSequence): The list of pulses to be repeated.
            num_repetitions (:
            mt_pulse_index (int|None): Index of the MT pulse in the kernel.
                If None, use the pulse with zero carrier frequency.
            *args:
            **kwargs:

        Returns:

        """
        SteadyState.__init__(self, *args, **kwargs)
        idx = self.get_unique_pulses(('MagnetizationPreparation',))
        if hasattr(self, 'idx'):
            self._idx.update(idx)
        self.freqs = freqs
        self.fas = fas

    # -----------------------------------
    def signal(
            self,
            spin_model,
            *args,
            **kwargs):
        base_p_ops = self.propagators(spin_model, *args, **kwargs)
        te_p_ops = [
            Delay(t_d).propagator(spin_model, *args, **kwargs)
            for t_d in self._pre_post_delays(self.te, self._t_ro, self._t_pexc)]
        mt_pulse = self.pulses[self._idx['MagnetizationPreparation']]
        f_c = mt_pulse.carrier_freq
        mt_p_ops = [
            mt_pulse.set_carrier_freq(f_c + f).set_flip_angle(fa).propagator(
                spin_model, *args, **kwargs)
            for f in self.freqs for fa in self.fas]
        central_p_ops_list = [
            self._p_ops_reorder(
                self._p_ops_subst(
                    base_p_ops, self._idx['MagnetizationPreparation'],
                    mt_p_op),
                self._idx['ReadOut'])
            for mt_p_op in mt_p_ops]
        s_arr = np.array([
            self._signal(
                spin_model,
                self._propagator(
                    [te_p_ops[0]] + central_p_ops + [te_p_ops[1]],
                    self.n_r))
            for central_p_ops in central_p_ops_list]).reshape(
            (len(self.freqs), len(self.fas)))
        return s_arr


# ======================================================================
class MultiMtVarMGESS(MultiGradEchoSteadyState):
    # -----------------------------------
    def __init__(
            self,
            preps,
            tes=None,
            tr=None,
            fa=None,
            *args,
            **kwargs):
        """

        Args:
            preps ():
            tes ():
            tr ():
            *args ():
            **kwargs ():
        """


        if tes is None and tr is not None:
            assert (all(
                [len(prep) == len(self.prep_labels) - 2 for prep in preps]))
        elif (tes is None) != (tr is None):
            assert (all(
                [len(prep) == len(self.prep_labels) - 1 for prep in preps]))
        else:
            assert (all([len(prep) == len(self.prep_labels) for prep in preps]))
        if tr is None:
            tr = preps[0][self.prep_labels.index('TR')]
        MultiGradEchoSteadyState.__init__(self, tes, tr, *args, **kwargs)
        idx = self.get_unique_pulses(('MagnetizationPreparation',))
        if hasattr(self, 'idx'):
            self._idx.update(idx)
        self.preps = preps

    # -----------------------------------
    @staticmethod
    def get_prep_labels():
        return 'PrepFrequency', 'PrepFlipAngle', 'FlipAngle', 'TR', 'TEs'

    # -----------------------------------
    @staticmethod
    def get_prep_units():
        return 'Hz', 'deg', 'deg', 's', 's'

    # -----------------------------------
    def signal(
            self,
            spin_model,
            *args,
            **kwargs):
        base_p_ops = self.propagators(spin_model, *args, **kwargs)
        unique_pre_post_delays = set([
            self._pre_post_delays(te, self._get_t_ro(self.duration, ),
                self._t_pexc)
            for df, mfa, fa, tr, tes in preps])
        unique_pre_post_p_ops = {
            t_d: Delay(t_d).propagator(spin_model, *args, **kwargs)
            for t_d in unique_pre_post_delays}
        mt_pulse = self.pulses[self._idx['MagnetizationPreparation']]
        f_c = mt_pulse.carrier_freq
        unique_mt = set([(df, mfa) for df, mfa, tr, tes in self.preps])
        unique_mt_p_ops = [
            mt_pulse.set_carrier_freq(f_c + df).set_flip_angle(fa).propagator(
                spin_model, *args, **kwargs)
            for df, mfa in unique_mt]

        central_p_ops_list = [
            self._p_ops_reorder(
                self._p_ops_subst(
                    base_p_ops, self._idx['MagnetizationPreparation'],
                    mt_p_op),
                self._idx['ReadOut'])
            for mt_p_op in mt_p_ops]
        s_arr = np.array([
            self._signal(
                spin_model,
                self._propagator(
                    [te_p_ops[0]] + central_p_ops + [te_p_ops[1]],
                    self.n_r))
            for central_p_ops in central_p_ops_list]).reshape(
            (len(self.freqs), len(self.fas)))
        return s_arr


# ======================================================================
elapsed(__file__[len(DIRS['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
