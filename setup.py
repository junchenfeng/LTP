from setuptools import setup

setup(
    name='LTP',
    version="0.0.1",
    packages=['LTP',
              'LTP/HMM',
              'LTP/MLC',],
    license='MIT',
    description='A python implementation of Learning Through Practices.',
    author='Junchen Feng',
    author_email='frankfeng.pku@gmail.com',
    install_requires=['numpy',
                      'scipy',
                      'cython', ],
    classifiers=[
                'Intended Audience :: Developers',
                'Programming Language :: Python :: 3.3',
                ],
        )
