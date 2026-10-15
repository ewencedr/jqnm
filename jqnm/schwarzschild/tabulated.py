""" Computing, loading, and storing tabulated Schwarzschild QNMs.

This module uses JAX for GPU acceleration and automatic differentiation.
"""

from __future__ import division, print_function, absolute_import

import logging
from pathlib import Path

import numpy as np

from ..angular import l_min
from .overtonesequence import SchwOvertoneSeq

# TODO some documentation here, better documentation throughout

def build_Schw_dict(*args, **kwargs):
    """ Function to build a dict of Schwarzschild QNMs.

    Loops over values of (s,l), using SchwOvertoneSeq to find
    sequences in n.

    TODO Documentation

    Parameters
    ----------
    s_arr: [int] [default: [-2, -1, 0]]
      Array of s values to run over.

    n_max: int [default: 20]
      Maximum overtone number to run over (inclusive).

    l_max: int [default: 20]
      Maximum angular harmonic number to run over (inclusive).

    tol: float [default: 1e-10]
      Tolerance to pass to SchwOvertoneSeq.

    Returns
    -------
    dict
      A dict with tuple keys (s,l,n).
      The value at d[s,l,n] is a tuple (omega, cf_err, n_frac)
      where omega is the frequency omega_{s,l,n}, cf_err is the
      estimated truncation error for the continued fraction, and
      n_frac is the depth of the continued fraction evaluation.

    """

    # TODO: Should we allow any other params to be customizable?
    s_arr   = kwargs.get('s_arr',   [-2, -1, 0])
    n_max   = kwargs.get('n_max',   20)
    l_max   = kwargs.get('l_max',   20)

    tol     = kwargs.get('tol',     1e-10)

    Schw_dict = {}
    Schw_err_dict = {}

    for s in s_arr:
        ls = np.arange(l_min(s,0),l_max+1)
        for l in ls:
            Schw_seq = SchwOvertoneSeq(s=s, l=l,
                                       n_max=n_max, tol=tol)
            try:
                Schw_seq.find_sequence()
            except:
                logging.warn("Failed at s={}, l={}".format(s, l))
            for n, (omega, cf_err, n_frac) in enumerate(zip(Schw_seq.omega,
                                                            Schw_seq.cf_err,
                                                            Schw_seq.n_frac)):
                Schw_dict[(s,l,n)] = (complex(omega),
                                      float(cf_err),
                                      int(n_frac))
                Schw_dict[(-s,l,n)] = Schw_dict[(s,l,n)]

    return Schw_dict

############################################################
def default_pickle_file():
    """Give the default path of the QNM dict file, `<dirname of this file>/data/Schw_dict.npz`.

    The file holds plain NumPy arrays (see :meth:`QNMDict.write_dict`), so
    loading it needs neither pickle nor the original `qnm` package.  The
    function keeps its historical name for compatibility.

    Returns
    -------
    Path object
      `<dirname of this file>/data/Schw_dict.npz`

    """

    return Path(__file__).parent.resolve() / 'data' / 'Schw_dict.npz'

############################################################

class QNMDict(object):
    """ Object for getting/holding/(pre-)computing Schwarzschild QNMs.

    This class uses the "borg" pattern, so the table of precomputed
    values will be shared amongst all instances of the class.  A set
    of precomputed QNMs can be loaded/stored from an `.npz` file with
    :meth:`load_dict()` and :meth:`write_dict()`.  The main interface
    is via the special :meth:`__call__()` method which is invoked via
    `object(s,l,n)`.  If the QNM labeled by (s,l,n) has already been
    computed, it will be returned.  Otherwise we try to compute it and
    then return it.

    Attributes
    ----------
    seq_dict: dict
      Keys are tuples (s,l) which label an overtone sequence of QNMs.
      Values are instances of
      :class:`jqnm.schwarzschild.overtonesequence.SchwOvertoneSeq`.

    loaded_from_disk: bool
      Whether or not any modes have been loaded from disk

    Parameters
    ----------
    init: bool, optional [default: False]
      Whether or not to call :meth:`load_dict()` when initializing
      this instance.

    dict_pickle_file: string or Path, optional [default: from default_pickle_file()]
      Path to the `.npz` file that holds precomputed QNMs. If the value is
      None, get the default from :meth:`default_pickle_file()`.
    """

    # Borg pattern, the QNM table will be shared among all instances
    _shared_state = {}

    loaded_from_disk = False

    # TODO: should we take a dict to pass to SchwOvertoneSeq?
    def __init__(self, init=False, dict_pickle_file=default_pickle_file()):

        self.__dict__ = self._shared_state

        if (not hasattr(self, 'seq_dict')):
            # First!
            self.seq_dict = {}

        if (init):
            self.load_dict(dict_pickle_file=dict_pickle_file)

    def load_dict(self, dict_pickle_file=default_pickle_file()):
        """ Load a Schw QNM dict from disk.

        Parameters
        ----------
        dict_pickle_file: string or Path [default: from default_pickle_file()]
          `.npz` file written by :meth:`write_dict()`.

        """

        # convert string to Path (does nothing to Path object)
        dict_pickle_file = Path(dict_pickle_file)

        try:
            with np.load(dict_pickle_file, allow_pickle=False) as data:
                logging.info('Loading Schw QNM dict from file {}'.format(dict_pickle_file))
                bounds = np.concatenate([[0], np.cumsum(data['seq_len'])])
                for i, (s, l) in enumerate(zip(data['s'], data['l'])):
                    lo, hi = bounds[i], bounds[i + 1]
                    seq = SchwOvertoneSeq(s=int(s), l=int(l), n_max=int(hi - lo - 1))
                    seq.omega = data['omega'][lo:hi]
                    seq.cf_err = data['cf_err'][lo:hi]
                    seq.n_frac = data['n_frac'][lo:hi]
                    seq.n = list(range(hi - lo))
                    self.seq_dict[(int(s), int(l))] = seq
            self.loaded_from_disk = True
        except (OSError, KeyError, ValueError) as e:
            logging.info("Could not load Schw QNM dict from file {}: {}".format(dict_pickle_file, e))
            self.loaded_from_disk = False

        return

    def write_dict(self, dict_pickle_file=default_pickle_file()):
        """Write the current state of the QNM dict to disk.

        The file is an `.npz` archive of plain arrays: `s`, `l` and
        `seq_len` per overtone sequence, and `omega`, `cf_err`, `n_frac`
        concatenated over all sequences in that order.

        Parameters
        ----------
        dict_pickle_file: string [default: from default_pickle_file()]
          Filename for writing the dict of Schwarzschild QNMs
        """

        # convert string to Path (does nothing to Path object)
        dict_pickle_file = Path(dict_pickle_file)

        pickle_dir = dict_pickle_file.parent.resolve()
        pickle_dir.mkdir(parents=True, exist_ok=True)

        keys = sorted(self.seq_dict.keys())
        seqs = [self.seq_dict[k] for k in keys]

        try:
            with dict_pickle_file.open('wb') as handle:
                logging.info("Writing Schw QNM dict to file {}".format(dict_pickle_file))
                np.savez(
                    handle,
                    s=np.array([seq.s for seq in seqs], dtype=int),
                    l=np.array([seq.l for seq in seqs], dtype=int),
                    seq_len=np.array([len(seq.omega) for seq in seqs], dtype=int),
                    omega=np.concatenate([seq.omega for seq in seqs]).astype(complex),
                    cf_err=np.concatenate([seq.cf_err for seq in seqs]).astype(float),
                    n_frac=np.concatenate([seq.n_frac for seq in seqs]).astype(float),
                )
        except OSError:
            logging.warning("Could not write Schw QNM dict to file {}".format(dict_pickle_file))

        return

    def __call__(self, s, l, n):
        """Get the Schwarzschild QNM labeled by (s,l,n).

        If the QNM has already been computed, immediately return that
        value.  If the (s,l) sequence is in the dict, but n has not
        been computed, extend the sequence to n and return the QNM.
        If the (s,l) sequence is not already in the dict, add it to
        the dict out to overtone n.

        Parameters
        ----------
        s: int
          Spin weight of the field.

        l: int
          Multipole number of the QNM.

        n: int
          Overtone number of the QNM.

        Returns
        -------
        (complex, double, int)
          The complex value is the QNM frequency.  The double is the
          estimated truncation error for the continued fraction.  The
          int is the depth of the continued fraction evaluation.

        Raises
        ------
        TODO

        """

        # TODO Check values of (s,l)

        if ((s,l) not in self.seq_dict.keys()):
            self.seq_dict[(s,l)] = SchwOvertoneSeq(s=s, l=l, n_max=n)

        seq = self.seq_dict[(s,l)]

        # This is idempotent, no danger in "extending" if the data
        # have already been computed. Go to n+2 to try to guard
        # against skipping an overtone in the sequence.
        # This can raise an exception.
        seq.extend(n_max=n+2)

        return (seq.omega[n], seq.cf_err[n], int(seq.n_frac[n]))
