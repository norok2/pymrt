#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup instructions.

See: https://packaging.python.org/en/latest/distributing.html
"""

# ======================================================================
# :: Future Imports (for Python 2)
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
# from __future__ import unicode_literals

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import re  # Regular expression operations
from codecs import open  # use a consistent encoding (in Python 2)

# ======================================================================
# :: Choice of the setup tools
from setuptools import setup
from setuptools import find_packages

# ======================================================================
# project specific variables
VERSION_FILEPATH = 'pymrt/__init__.py'
README_FILEPATH = 'README.rst'

# get the working directory for the setup script
CWD = os.path.realpath(os.path.dirname(__file__))

# get the long description from the README file
with open(os.path.join(CWD, README_FILEPATH), encoding='utf-8') as readme_file:
    LONG_DESCRIPTION_TEXT = readme_file.read()


# ======================================================================
def fix_version(
        version=None,
        source_filepath=VERSION_FILEPATH):
    """
    Fix version in source code.

    Args:
        version (str): version to be used for fixing the source code
        source_filepath (str): Path to file where __version__ is located

    Returns:
        version (str): the actual version text used
    """
    if version is None:
        import setuptools_scm

        version = setuptools_scm.get_version()
    with open(source_filepath, 'r') as src_file:
        src_str = src_file.read().decode('utf-8')
        src_str = re.sub(
            r"__version__ = '.*'",
            "__version__ = '{}'".format(version),
            src_str, flags=re.UNICODE)

    with open(source_filepath, 'w') as src_file:
        src_file.write(src_str.encode('utf-8'))

    return version


# ======================================================================
# :: call the setup tool
setup(
    name='pymrt',

    description='Data analysis for quantitative MRI',
    long_description=LONG_DESCRIPTION_TEXT,

    version=fix_version(),

    url='https://bitbucket.org/norok2/pymrt',

    author='Riccardo Metere',
    author_email='rick@metere.it',

    license='GPLv3+',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Environment :: Console',
        'Environment :: X11 Applications',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: System Administrators',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Utilities',

        'Operating System :: POSIX',
        'Natural Language :: English',

        'License :: OSI Approved :: GNU General Public License v3 or later'
        ' (GPLv3+)',

        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],

    keywords=('quantitative', 'MRI', 'qMRI', 'MT', 'susceptibility',
              'relaxometry', 'qMT', 'neurophysics', 'neurology', 'physics',
              'imaging', 'data', 'analysis'),

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=[
        'numpy',
        'scipy',
        'sympy',
        'nibabel',
        'matplotlib',
        'pyparsing',
        'numeral',
    ],

    setup_requires=[
        'setuptools',
        'setuptools_scm'
    ],

    extras_require={
        'blessed': 'blessed',
    },

    package_data={
        'pymrt': ['sequences/pulses/*.csv', ],
    },

    # data_files=[('my_data', ['data/data_file'])],

    entry_points={
        'console_scripts': [
            'pymrt_compute=scripts_cli.compute:main',
            'pymrt_correlate=scripts_cli.correlate:main',

            'pymrt_calc_mask=scripts_cli.calc_mask:main',
            'pymrt_geom_phantom=scripts_cli.geom_phantom:main',
            'pymrt_ernst_calc=scripts_cli.ernst_calc:main',
            'pymrt_unwrap=scripts_cli.unwrap:main',

            'pymrt_extract_nifti_bruker=scripts_cli.extract_nifti_bruker:main',
        ],

        'gui_scripts': [
            'pymrt_flash=scripts_gui.flash_sim_optim:main',
            'pymrt_mp2rage=scripts_gui.mp2rage_sim_optim:main',
        ],
    },
)
