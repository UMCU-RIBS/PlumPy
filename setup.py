from setuptools import setup, find_packages

setup(
    name='plumpy',
    version='0.1.0',    
    description='Python package for analyzing ECoG implant data',
    url='https://github.com/UMCU-RIBS/PlumPy',
    author='Julia Berezutskaya',
    author_email='y.berezutskaya@umcutrecht.nl',
    license='MIT',
    packages=find_packages(),
    install_requires=['matplotlib',
                      'numpy',
                      'mne',
                      'optuna',
                      'pandas',
                      'PyYAML',
                      'scikit_learn',                 
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.11',
    ],
)
