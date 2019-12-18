#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of frequently-used imports."""

# ======================================================================
# :: Future Imports
from __future__ import(
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports (updated to Python 3.7)

# # : Text Processing Services
# import string  # Common string operations
# import re  # Regular expression operations
# import difflib  # Helpers for computing deltas
# import textwrap  # Text wrapping and filling
# import unicodedata  # Unicode Database
# import stringprep  # Internet String Preparation
# import readline  # GNU readline interface
# import rlcompleter  # Completion function for GNU readline
#
# # : Binary Data Services
# import struct  # Interpret bytes as packed binary data
# import codecs  # Codec registry and base classes
#
# # : Data Types
# import datetime  # Basic date and time types
# import calendar  # Dasic date and time types
# import collections  # Container datatypes
# import collections.abc  # Abstract Base Classes for Containers
# import heapq  # Heap queue algorithm
# import bisect  # Array bisection algorithm
# import array  # Efficient arrays of numeric values
# import weakref  # Weak references
# import types  # Dynamic type creation and names for built-in types
# import copy  # Shallow and deep copy operations
# import pprint  # Data pretty printer
# import reprlib  # Alternate repr() implementation
# import enum  # Support for enumerations
#
# # : Numeric and Mathematical Modules
# import numbers  # Numeric abstract base classes
# import math  # Mathematical functions
# import cmath  # Mathematical functions for complex numbers
# import decimal  # Decimal fixed point and floating point arithmetic
# import fractions  # Rational numbers
# import random  # Generate pseudo-random numbers
# import statistics  # Mathematical statistics functions
#
# # : Functional Programming Modules
# import itertools  # Functions creating iterators for efficient looping
# import functools  # Higher-order functions and operations on callable objects
# import operator  # Standard operators as functions
#
# # : File and Directory Access
# import pathlib  # Object-oriented filesystem paths
# import os.path  # Common pathname manipulations
# import fileinput  # Iterate over lines from multiple input streams
# import stat  # Interpreting stat() results
# import filecmp  # File and Directory Comparisons
# import tempfile  # Generate temporary files and directories
# import glob  # Unix style pathname pattern expansion
# import fnmatch  # Unix filename pattern matching
# import linecache  # Random access to text lines
# import shutil  # High-level file operations
# import macpath  # Mac OS 9 path manipulation functions
#
# # : Data Persistence
# import pickle  # Python object serialization
# import copyreg  # Register pickle support functions
# import shelve  # Python object persistence
# import marshal  # Internal Python object serialization
# import dbm  # Interfaces to Unix “databases”
# import sqlite3  # DB-API 2.0 interface for SQLite databases
#
# # : Data Compression and Archiving
# import zlib  # Compression compatible with gzip
# import gzip  # Support for gzip files
# import bz2  # Support for bzip2 compression
# import lzma  # Compression using the LZMA algorithm
# import zipfile  # Work with ZIP archives
# import tarfile  # Read and write tar archive files
#
# # : File Formats
# import csv  # CSV File Reading and Writing
# import configparser  # Configuration file parser
# import netrc  # netrc file processing
# import xdrlib  # Encode and decode XDR data
# import plistlib  # Generate and parse Mac OS X .plist files
#
# # : Cryptographic Services
# import hashlib  # Secure hashes and message digests
# import hmac  # Keyed-Hashing for Message Authentication
# import secrets  # Generate secure random numbers for managing secrets
#
# # : Generic Operating System Services
# import os  # Miscellaneous operating system interfaces
# import io  # Core tools for working with streams
# import time  # Time access and conversions
# import argparse  # Parser for command-line options, arguments and sub-commands
# import getopt  # C-style parser for command line options
# import logging  # Logging facility for Python
# import logging.config  # Logging configuration
# import logging.handlers  # Logging handlers
# import getpass  # Portable password input
# import curses  # Terminal handling for character-cell displays
# import curses.textpad  # Text input widget for curses programs
# import curses.ascii  # Utilities for ASCII characters
# import curses.panel  # A panel stack extension for curses
# import platform  # Access to underlying platform’s identifying data
# import errno  # Standard errno system symbols
# import ctypes  # A foreign function library for Python
#
# # : Concurrent Execution
# import threading  # Thread-based parallelism
# import multiprocessing  # Process-based parallelism
#
# import concurrent.futures  # Launching parallel tasks
# import subprocess  # Subprocess management
# import sched  # Event scheduler
# import queue  # A synchronized queue class
# import _thread  # Low-level threading API
# import _dummy_thread  # Drop-in replacement for the _thread module
# import dummy_threading  # Drop-in replacement for the threading module
#
# # : Context Variables
# import contextvars  # Context Variables
#
# # : Interprocess Communication and Networking
# import socket  # Low-level networking interface
# import ssl  # TLS/SSL wrapper for socket objects
# import select  # Waiting for I/O completion
# import selectors  # High-level I/O multiplexing
# import asyncio  # Asynchronous I/O, event loop, coroutines and tasks
# import asyncore  # Asynchronous socket handler
# import asynchat  # Asynchronous socket command/response handler
# import signal  # Set handlers for asynchronous events
# import mmap  # Memory-mapped file support
#
# # : Internet Data Handling
# import email  # An email and MIME handling package
# import json  # JSON encoder and decoder
# import mailcap  # Mailcap file handling
# import mailbox  # Manipulate mailboxes in various formats
# import mimetypes  # Map filenames to MIME types
# import base64  # Base16, Base32, Base64, Base85 Data Encodings
# import binhex  # Encode and decode binhex4 files
# import binascii  # Convert between binary and ASCII
# import quopri  # Encode and decode MIME quoted-printable data
# import uu  # Encode and decode uuencode files
#
# # : Structured Markup Processing Tools
# import html  # HyperText Markup Language support
# import html.parser  # Simple HTML and XHTML parser
# import html.entities  # Definitions of HTML general entities
# import xml  # XML Processing Modules
# import xml.etree.ElementTree  # The ElementTree XML API
# import xml.dom  # The Document Object Model API
# import xml.dom.minidom  # Minimal DOM implementation
# import xml.dom.pulldom  # Support for building partial DOM trees
# import xml.sax  # Support for SAX2 parsers
# import xml.sax.handler  # Base classes for SAX handlers
# import xml.sax.saxutils  # SAX Utilities
# import xml.sax.xmlreader  # Interface for XML parsers
# import xml.parsers.expat  # Fast XML parsing using Expat
#
# # : Internet Protocols and Support
# import webbrowser  # Convenient Web-browser controller
# import cgi  # Common Gateway Interface support
# import cgitb  # Traceback manager for CGI scripts
# import wsgiref  # WSGI Utilities and Reference Implementation
# import urllib  # URL handling modules
# import urllib.request  # Extensible library for opening URLs
# import urllib.response  # Response classes used by urllib
# import urllib.parse  # Parse URLs into components
# import urllib.error  # Exception classes raised by urllib.request
# import urllib.robotparser  # Parser for robots.txt
# import http  # HTTP modules
# import http.client  # HTTP protocol client
# import ftplib  # FTP protocol client
# import poplib  # POP3 protocol client
# import imaplib  # IMAP4 protocol client
# import nntplib  # NNTP protocol client
# import smtplib  # SMTP protocol client
# import smtpd  # SMTP Server
# import telnetlib  # Telnet client
# import uuid  # UUID objects according to RFC 4122
# import socketserver  # A framework for network servers
# import http.server  # HTTP servers
# import http.cookies  # HTTP state management
# import http.cookiejar  # Cookie handling for HTTP clients
# import xmlrpc  # XMLRPC server and client modules
# import xmlrpc.client  # XML-RPC client access
# import xmlrpc.server  # Basic XML-RPC servers
# import ipaddress  # IPv4/IPv6 manipulation library
#
# # : Multimedia Services
# import audioop  # Manipulate raw audio data
# import aifc  # Read and write AIFF and AIFC files
# import sunau  # Read and write Sun AU files
# import wave  # Read and write WAV files
# import chunk  # Read IFF chunked data
# import colorsys  # Conversions between color systems
# import imghdr  # Determine the type of an image
# import sndhdr  # Determine type of sound file
# import ossaudiodev  # Access to OSS-compatible audio devices
#
# # : Internationalization
# import gettext  # Multilingual internationalization services
# import locale  # Internationalization services
#
# # : Program Frameworks
# import turtle  # Turtle graphics
# import cmd  # Support for line-oriented command interpreters
# import shlex  # Simple lexical analysis
#
# # : Graphical User Interfaces with Tk
# import tkinter  # Python interface to Tcl/Tk
# import tkinter.ttk  # Tk themed widgets
# import tkinter.tix  # Extension widgets for Tk
# import tkinter.scrolledtext  # Scrolled Text Widget
#
#
# # : Development Tools
# import typing  # Support for type hints
# import pydoc  # Documentation generator and online help system
# import doctest  # Test interactive Python examples
# import unittest  # Unit testing framework
# import unittest.mock  # mock object library
# import unittest.mock  # getting started
# import lib2to3  # Automated Python 2 to 3 code translation
# import test  # Regression tests package for Python
# import test.support  # Utilities for the Python test suite
# import test.support.script_helper  # Utilities for the Python execution tests
#
# # : Debugging and Profiling
# import bdb  # Debugger framework
# import faulthandler  # Dump the Python traceback
# import pdb  # The Python Debugger
# import timeit  # Measure execution time of small code snippets
# import trace  # Trace or track Python statement execution
# import tracemalloc  # Trace memory allocations
#
# # : Software Packaging and Distribution
# import distutils  # Building and installing Python modules
# import ensurepip  # Bootstrapping the pip installer
# import venv  # Creation of virtual environments
# import zipapp  # Manage executable python zip archives
#
# # : Python Runtime Services
# import sys  # System-specific parameters and functions
# import sysconfig  # Provide access to Python’s configuration information
# import builtins  # Built-in objects
# import __main__  # Top-level script environment
# import warnings  # Warning control
# import dataclasses  # Data Classes
# import contextlib  # Utilities for with-statement contexts
# import abc  # Abstract Base Classes
# import atexit  # Exit handlers
# import traceback  # Print or retrieve a stack traceback
# import __future__  # Future statement definitions
# import gc  # Garbage Collector interface
# import inspect  # Inspect live objects
# import site  # Site-specific configuration hook
#
# # : Custom Python Interpreters
# import code  # Interpreter base classes
# import codeop  # Compile Python code
#
# # : Importing Modules
# import zipimport  # Import modules from Zip archives
# import pkgutil  # Package extension utility
# import modulefinder  # Find modules used by a script
# import runpy  # Locating and executing Python modules
# import importlib  # The implementation of import
#
# # : Python Language Services
# import parser  # Access Python parse trees
# import ast  # Abstract Syntax Trees
# import symtable  # Access to the compiler’s symbol tables
# import symbol  # Constants used with Python parse trees
# import token  # Constants used with Python parse trees
# import keyword  # Testing for Python keywords
# import tokenize  # Tokenizer for Python source
# import tabnanny  # Detection of ambiguous indentation
# import pyclbr  # Python class browser support
# import py_compile  # Compile Python source files
# import compileall  # Byte-compile Python libraries
# import dis  # Disassembler for Python bytecode
# import pickletools  # Tools for pickle developers
#
# # : Miscellaneous Services
# import formatter  # Generic output formatting
#
# # : MS Windows Specific Services
# import msilib  # Read and write Microsoft Installer files
# import msvcrt  # Useful routines from the MS VC++ runtime
# import winreg  # Windows registry access
# import winsound  # Sound-playing interface for Windows
#
# # : Unix Specific Services
# import posix  # The most common POSIX system calls
# import pwd  # The password database
# import spwd  # The shadow password database
# import grp  # The group database
# import crypt  # Function to check Unix passwords
# import termios  # POSIX style tty control
# import tty  # Terminal control functions
# import pty  # Pseudo-terminal utilities
# import fcntl  # The fcntl and ioctl system calls
# import pipes  # Interface to shell pipelines
# import resource  # Resource usage information
# import nis  # Interface to Sun’s NIS (Yellow Pages)
# import syslog  # Unix syslog library routines
#
# # : Superseded Modules
# import optparse  # Parser for command line options
# import imp  # Access the import internals

# :: Python 2 support
# # Configuration file parser
# try:
#     import configparser
# except ImportError:
#     import ConfigParser as configparser

# :: External Imports
# import numpy as np  # NumPy (multidimensional numerical arrays library)
# import scipy as sp  # SciPy (signal and image processing library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import pandas as pd  # pandas (Python Data Analysis Library)
# import seaborn as sns  # Seaborn: statistical data visualization
# import PIL  # Python Image Library (image manipulation toolkit)

# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
# import mpl_toolkits.mplot3d as mpl3  # Matplotlib's 3D support
#
# import scipy.io  # SciPy: Input and output
# import scipy.optimize  # SciPy: Optimization
# import scipy.integrate  # SciPy: Integration
# import scipy.interpolate  # SciPy: Interpolation
# import scipy.constants  # SciPy: Constants
# import scipy.ndimage  # SciPy: Multidimensional image processing
# import scipy.linalg  # SciPy: Linear Algebra
# import scipy.stats  # SciPy: Statistical functions
# import scipy.misc  # SciPy: Miscellaneous routines
# import scipy.signal  # SciPy: Signal Processing

# import sympy.mpmath  # SymPy: Function approximation

# :: Local Imports
# import pymrt as mrt  # Python Magnetic Resonance Tools: the multi-tool of MRI.
# import pymrt.utils
# import pymrt.naming
# import raster_geometry  # Create/manipulate N-dim raster geometric shapes.
# import pymrt.plot
# import pymrt.registration
# import pymrt.segmentation
# import pymrt.computation
# import pymrt.correlation
# import pymrt.input_output
# import pymrt.sequences
# import pymrt.extras

# from pymrt.sequences import *
# from pymrt.extras import *
# from pymrt.recipes import *

# from pymrt import INFO, PATH, MY_GREETINGS
# from pymrt import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
# from pymrt import elapsed, report
# from pymrt import msg, dbg, fmt, fmtm
