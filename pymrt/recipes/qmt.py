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
        if hasattr(self, '_idx'):
            self._idx.update(idx)
        else:
            self._idx = idx
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
            mfas,
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
        if hasattr(self, '_idx'):
            self._idx.update(idx)
        else:
            self._idx = idx
        self.freqs = freqs
        self.fas = mfas

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
        MultiGradEchoSteadyState.__init__(self, None, None, *args, **kwargs)
        # fix preps
        if fa is None:
            fa = self.pulses[self._idx['PulseExc']].flip_angle
        len_labels = len(self.get_prep_labels())
        self.preps = []
        optional_preps = (
            (fa, 'FlipAngle', False),
            (tr, 'TR', False),
            (tes, 'TEs', True))
        for prep in preps:
            prep = list(prep) + [None] * (len_labels - len(prep))
            for param, label, is_seq in optional_preps:
                i = self.get_prep_labels().index(label)
                if param is not None and prep[i] is None:
                    prep[i] = param
                if is_seq:
                    prep[i] = mrt.utils.auto_repeat(prep[i], 1, False, False)
            assert (all([prep_val is not None for prep_val in prep]))
            self.preps.append(prep)

        idx = self.get_unique_pulses(('MagnetizationPreparation',))
        if hasattr(self, '_idx'):
            self._idx.update(idx)
        else:
            self._idx = idx

    # -----------------------------------
    @staticmethod
    def get_prep_labels():
        """

        Returns:

        """
        return 'PrepFrequency', 'PrepFlipAngle', 'FlipAngle', 'TR', 'TEs'

    # -----------------------------------
    @staticmethod
    def get_prep_units():
        """

        Returns:

        """
        return 'Hz', 'deg', 'deg', 's', 's'

    # -----------------------------------
    def _t_pre(self, te, tr):
        """

        Args:
            te ():
            tr ():

        Returns:

        """
        return self._pre_post_delays(
            te, self._get_t_ro(self.duration, tr), self._t_pexc)[0]

    # -----------------------------------
    def _t_post(self, te, tr):
        """

        Args:
            te ():
            tr ():

        Returns:

        """
        return self._pre_post_delays(
            te, self._get_t_ro(self.duration, tr), self._t_pexc)[1]

    # -----------------------------------
    def signal(
            self,
            spin_model,
            *args,
            **kwargs):
        """

        Args:
            spin_model ():
            *args ():
            **kwargs ():

        Returns:

        """
        base_p_ops = self.propagators(spin_model, *args, **kwargs)
        unique_pre_post_delays = set(mrt.utils.flatten([
            self._pre_post_delays(
                te, self._get_t_ro(self.duration, tr), self._t_pexc)
            for df, mfa, fa, tr, tes in self.preps for te in tes]))
        unique_pre_post_p_ops = {
            t_d: Delay(t_d).propagator(spin_model, *args, **kwargs)
            for t_d in unique_pre_post_delays}
        mt_pulse = self.pulses[self._idx['MagnetizationPreparation']]
        f_c = mt_pulse.carrier_freq
        unique_mt = set([(df, mfa) for df, mfa, fa, tr, tes in self.preps])
        unique_mt_p_ops = {
            (df, mfa):
                mt_pulse.set_carrier_freq(f_c + df).set_flip_angle(mfa).propagator(
                    spin_model, *args, **kwargs)
            for df, mfa in unique_mt}
        pexc_pulse = self.pulses[self._idx['PulseExc']]
        unique_pexc = set([fa for df, mfa, fa, tr, tes in self.preps])
        unique_pexc_p_ops = {
            fa: pexc_pulse.set_flip_angle(fa).propagator(
                spin_model, *args, **kwargs)
            for fa in unique_pexc}
        s_arr = np.array([
            self._signal(
                spin_model,
                self._propagator(
                    [unique_pre_post_p_ops[self._t_pre(te, tr)]] +
                    self._p_ops_reorder(
                        self._p_ops_substs(
                            base_p_ops, (
                                (self._idx['MagnetizationPreparation'],
                                 unique_mt_p_ops[(df, mfa)]),
                                (self._idx['PulseExc'],
                                 unique_pexc_p_ops[fa]),)),
                        self._idx['ReadOut']) +
                    [unique_pre_post_p_ops[self._t_post(te, tr)]],
                    self.n_r))
            for df, mfa, fa, tr, tes in self.preps for te in tes])
        return s_arr


# ======================================================================
elapsed(__file__[len(DIRS['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
