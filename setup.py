#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup instructions.

See: https://packaging.python.org/en/latest/distributing.html
"""

# ======================================================================
# :: Future Imports (for Python 2)
from __future__ import (
    division, absolute_import, print_function)
# BUG in setuptools if import unicode_literals

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
NAME = 'PyMRT'
VERSION_FILEPATH = os.path.join(NAME.lower(), '_version.py')
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

    def dummy_version():
        return '0.0.0.0'

    if version is None:
        try:
            from setuptools_scm import get_version
        except ImportError:
            get_version = dummy_version
        version = get_version()

    if not os.path.isfile(source_filepath):
        version_template = \
            '#!/usr/bin/env python3\n' \
            '# -*- coding: utf-8 -*-\n' \
            '"""Package version file."""\n' \
            '# This file is automatically generated by `fix_version()`' \
            ' in the setup script.\n' \
            '__version__ = \'{version}\'\n'.format(version=version)
        with open(source_filepath, 'wb') as io_file:
            io_file.write(version_template.encode('utf-8'))
    else:
        with open(source_filepath, 'rb') as io_file:
            source = io_file.read().decode('utf-8')
            source = re.sub(
                r"__version__ = '.*'",
                "__version__ = '{}'".format(version),
                source, flags=re.UNICODE)
        with open(source_filepath, 'wb') as io_file:
            io_file.write(source.encode('utf-8'))

    return version


# ======================================================================
# :: call the setup tool
setup(
    name=NAME.lower(),

    description='Python Magnetic Resonace Tools',
    long_description=LONG_DESCRIPTION_TEXT,

    version=fix_version(),

    url='https://bitbucket.org/norok2/' + NAME.lower(),

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

        'License :: OSI Approved :: '
        'GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
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
        'appdirs',
        'seaborn',
        'pywavelets',
        'h5py',
        'pillow',
        'numeral',
        'pytk',
        'numex',
        'flyingcircus',
        # 'numba',
    ],

    setup_requires=[
        'setuptools',
        'setuptools_scm',
        'appdirs',
    ],

    extras_require={
        'blessed': 'blessed',
        'weasyprint': 'weasyprint',
    },

    package_data={
        'pymrt': ['sequences/pulses/*.csv'],
        'resources': ['icon.*']
    },
    include_package_data=True,

    data_files=[('share/icons', ['artwork/pymrt_logo.svg'])],

    entry_points={
        'console_scripts': [
            'pymrt_batch_compute=scripts_cli.batch_compute:main',
            'pymrt_compute=scripts_cli.compute:main',
            'pymrt_correlate=scripts_cli.correlate:main',

            'pymrt_calc_mask=scripts_cli.calc_mask:main',
            'pymrt_gen_phantom=scripts_cli.gen_phantom:main',
            'pymrt_ernst_calc=scripts_cli.ernst_calc:main',
            'pymrt_unwrap=scripts_cli.unwrap:main',
            'pymrt_coil_combine=scripts_cli.coil_combine:main',

            'pymrt_extract_nifti_bruker=scripts_cli.extract_nifti_bruker:main',
        ],

        'gui_scripts': [
            'pymrt_mp2rage_t1=scripts_gui.mp2rage_t1_sensitivity_b1t:main',
            'pymrt_mp2rage_b1t=scripts_gui.mp2rage_b1t_sensitivity_t1:main',
            'pymrt_ernst_angle=scripts_gui.flash_ernst_angle:main',
        ],
    },
)

# ======================================================================
# :: create user directory
from pymrt import pkg_paths
