from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

here = path.abspath(path.dirname(__file__))
about = {}
with open(path.join(here, 'eland', '__version__.py'), 'r', 'utf-8') as f:
    exec(f.read(), about)

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=about['__url__'],
    maintainer=about['__maintainer__'],
    maintainer_email=about['__maintainer_email__'],
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='elastic eland pandas python',
    packages=['eland'],
    install_requires=[
        'elasticsearch>=7.0.5',
        'pandas==0.25.1',
        'matplotlib'
    ]
)
