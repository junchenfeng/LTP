from setuptools import setup

setup(
    name='pyMLC',
    version="0.0.1",
    packages=['pyMLC',
              'pyMLC/utl',
              'pyMLC/solver',],
    license='MIT',
    description='A python implementation of Mixture Learning Curve model.',
    author='Junchen Feng',
    author_email='frankfeng.pku@gmail.com',
    install_requires=['numpy',
                      'scipy',
                      'cython', ],
    classifiers=[
                'Intended Audience :: Developers',
                'Programming Language :: Python :: 2.7',
                ],
        )
