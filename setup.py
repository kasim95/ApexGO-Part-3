from setuptools import setup
from setuptools import find_packages

setup(name='ApexGO_BOT',
      version='1',
      description='481 Project',
      url='https://github.com/kasim95/ApexGO-Part-3',
      install_requires=[
            'numpy<=1.14.5',
            'tensorflow>=1.12.1',
            'keras==2.2.5',
            'gomill',
            'Flask>=0.10.1',
            'Flask-Cors',
            'future',
            'h5py',
            'six'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
