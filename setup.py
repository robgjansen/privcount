'''
Created on Dec 12, 2015

@author: rob
'''

#!/usr/bin/env python

from distutils.core import setup

setup(name='PrivCount',
      version='0.2.0',
      description='Safely gather and aggregate Tor statistics',
      author='Rob Jansen',
      packages=['privcount'],
      scripts=['privcount/tools/privcount'],
      # allow other packages to depend on "privcount [plot]"
      extras_require={
        'plot':  ['matplotlib', 'numpy']
      }
     )
