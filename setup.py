from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='eland',
      version='0.1',
      description='Python elasticsearch client to analyse, explore and manipulate data that resides in elasticsearch',
      url='http://github.com/elastic/eland',
      author='Stephen Dodson',
      author_email='sjd171@gmail.com',
      license='ELASTIC LICENSE',
      packages=['eland'],
      install_requires=[
          'elasticsearch',
          'elasticsearch_dsl',
          'pandas',
          'modin',
          'py'
      ],
      zip_safe=False)
