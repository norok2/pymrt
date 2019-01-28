=======================================
PyMRT - Python Magnetic Resonance Tools
=======================================

**PyMRT** - Python Magnetic Resonance Tools

.. code::

     ____        __  __ ____ _____
    |  _ \ _   _|  \/  |  _ \_   _|
    | |_) | | | | |\/| | |_) || |
    |  __/| |_| | |  | |  _ < | |
    |_|    \__, |_|  |_|_| \_\|_|
           |___/

Overview
--------
This software provides a support Python library and auxiliary console-based
tools to perform common tasks for Magnetic Resonance Imaging (MRI).
The aim is to be the Swiss Army knife of MRI.

At present, the following features are best supported:

- generic tools for image analysis from an MRI perspective
- data analysis for quantitative MRI experiments

On top of this, additional effort is currently being put in the following
areas:

- image reconstruction and related features (e.g. coil combination, etc.)

It is relatively easy to extend and users are encouraged to tweak with it.

As a result of the code maturity, some of the library components may undergo
(eventually heavy) refactoring, although this is currently unexpected.


Releases information are available through ``NEWS.rst``.

For a more comprehensive list of changes see ``CHANGELOG.rst``.


Installation
------------
The recommended way of installing the software is through
`PyPI <https://pypi.python.org/pypi/pymrt>`_:

.. code:: shell

    $ pip install pymrt

Alternatively, you can clone the source repository from
`Bitbucket <https://bitbucket.org/norok2/pymrt>`_:

.. code:: shell

    $ mkdir pymrt
    $ cd pymrt
    $ git clone git@bitbucket.org:norok2/pymrt.git
    $ python setup.py install

For more details see also ``INSTALL.rst``.

License
-------
This work is licensed through the terms and conditions of the
`GPLv3+ <http://www.gnu.org/licenses/gpl-3.0.html>`_

The use of this software for scientific purpose leading to a publication
should be acknowledged through citation of the following reference:

**Metere, R., Möller, H.E., 2017. PyMRT and DCMPI: Two New Python Packages for MRI Data Analysis, #3816: Proceedings of the 25th Annual Meeting & Exhibition of the International Society for Magnetic Resonance in Medicine (ISMRM), Honolulu, Hawaii, USA.**


Acknowledgements
----------------
This software originated as part of the Ph.D. work of Riccardo Metere at the
`Max Planck Institute for Human Cognitive and Brain Sciences
<http://www.cbs.mpg.de>`_ and the `University of Leipzig
<http://www.uni-leipzig.de>`_, and has been constantly expanded from there.

For a complete list of authors please see ``AUTHORS.rst``.

People who have influenced this work are acknowledged in ``THANKS.rst``.

This work was partly funded by the European Union
through the Seventh Framework Programme Marie Curie Actions
via the "Ultra-High Field Magnetic Resonance Imaging: HiMR"
Initial Training Network (FP7-PEOPLE-2012-ITN-316716).
