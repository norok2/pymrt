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
VERSION_SOURCE_FILEPATH = 'mri_tools/__init__.py'
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
    name='mri_tools',

    description='DICOM Preprocessing Interface.',
    long_description=LONG_DESCRIPTION_TEXT,

    version=fix_version(),

    url='https://bitbucket.org/norok2/dcmpi',

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
        'matplotlib', ],

    package_data={
        'dcmpi': ['report_templates/*.html', ],
    },

    # data_files=[('my_data', ['data/data_file'])],

    entry_points={
        'console_scripts': [
            'dcmpi_gui=dcmpi.dcmpi_gui:main',
            'dcmpi=dcmpi.dcmpi_cli:main',

            'dcmpi_monitor_folder=dcmpi.dcmpi_monitor_folder:main',

            'dcmpi__import_sources=dcmpi.import_sources:main',
            'dcmpi__sorting=dcmpi.sorting:main',
            'dcmpi__get_info=dcmpi.get_info:main',
            'dcmpi__get_meta=dcmpi.get_meta:main',
            'dcmpi__get_nifti=dcmpi.get_nifti:main',
            'dcmpi__get_prot=dcmpi.get_prot:main',
            'dcmpi__report=dcmpi.report:main',
            'dcmpi__backup=dcmpi.backup:main',
        ],
    },
)
