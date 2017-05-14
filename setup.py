from setuptools import setup

setup(
    name='LTP',
    version="0.0.4",
    packages=[
                'LTP',
                'LTP.HMM'
            ],
    package_dir = {
        'LTP':'src/LTP',
        'LTP.HMM':'src/LTP/HMM'
        },
    license='MIT',
    description='A python implementation of Learning Through Practices.',
    author='Junchen Feng',
    author_email='frankfeng.pku@gmail.com',
    install_requires=['numpy',
                      'cython',
                      'tqdm',
                      'joblib'],
    classifiers=[
                'Intended Audience :: Developers',
                'Programming Language :: Python :: 3.6',
                ],
        )
