#!python
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
VERSION_SOURCE_FILEPATH = 'pymrt/__init__.py'
README_SOURCE_FILE = 'README.rst'

# get the working directory for the setup script
CWD = os.path.realpath(os.path.dirname(__file__))

# get the long description from the README file
with open(os.path.join(CWD, README_SOURCE_FILE),
          encoding='utf-8') as readme_file:
    LONG_DESCRIPTION_TEXT = readme_file.read()


# ======================================================================
def fix_version(
        version=None,
        source_filepath=VERSION_SOURCE_FILEPATH):
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
        src_str = src_file.read()
        src_str = re.sub(
            r"__version__ = '.*'",
            "__version__ = '{}'".format(version),
            src_str)

    with open(source_filepath, 'w') as src_file:
        src_file.write(src_str)

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
        'Development Status :: 4 - Beta',

        'Intended Audience :: System Administrators',

        'Topic :: System :: Shells',
        'Topic :: System :: Systems Administration',
        'Topic :: System :: Filesystems',
        'Topic :: System :: Monitoring',
        'Topic :: Utilities',

        'Operating System :: POSIX',

        'License :: OSI Approved :: GNU General Public License v3 or later'
        ' (GPLv3+)',

        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='dicom dcm preprocessing',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=[
        'numpy',
        'scipy',
        'sympy',
        'nibabel',
        'matplotlib',
        'pyparsing',
    ],

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

            'pymrt_extract_nifti_bruker=scripts_cli.extract_nifti_bruker:main',
        ],

        'gui_scripts': [
            'pymrt_ernst_calc=scripts_gui.ernst_angle_calculator:main',
            'pymrt_mp2rage=scripts_gui.mp2rage_sim_optim:main',
        ],
    },
)
