#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='Conic',
    version='0.0.1',
    description='Conic section plot',
    author='Eivind Fonn',
    author_email='eivind.fonn@sintef.no',
    license='GPL3',
    url='https://github.com/TheBB/conic',
    py_modules=['conic'],
    install_requires=['click', 'matplotlib', 'numpy', 'scipy'],
    entry_points={
        'console_scripts': ['conic=conic.__main__:main'],
    },
)
