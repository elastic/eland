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
          'elasticsearch>=7.0.5',
          'pandas==0.25.1'
      ],
      zip_safe=False)
