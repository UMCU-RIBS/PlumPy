from setuptools import setup

setup(
    name='plumpy',
    version='0.1.0',    
    description='Python package for analyzing ECoG implant data',
    url='https://github.com/UMCU-RIBS/PlumPy',
    author='Julia Berezutskaya',
    author_email='y.berezutskaya@umcutrecht.nl',
    license='MIT',
    packages=['plumpy'],
    install_requires=['scikit-learn',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.11',
    ],
)
