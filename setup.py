import setuptools

from foundry import __version__

setuptools.setup(
    name='foundry',
    version=__version__,
    author='Jacob Dink',
    packages=setuptools.find_packages(include='foundry.*'),
    zip_safe=False,
    install_requires=[
        'pandas>=1.3',
        'torch>=1.9',
        'tqdm',
        'scikit-learn>=1.0',
        'scipy>=1.7',
        'tenacity>=8.0'
    ],
    python_requires='>=3.7',
    extras_require={
        'dev': [
            'jupytext',
            # pinning these is needed because notebook>6 isn't working with jupytext
            'notebook==6.5.4',
            'traitlets==5.9',
            'pytest',
            'requests',
            'plotnine',
        ],
        'docs': [
            'requests'
        ]
    }
)
