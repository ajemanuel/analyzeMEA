from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='analyzeMEA',
      version='0.1',
      description='Scripts for analyzing multielectrode array recordings',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/ajemanuel/analyzeMEA',
      author='Alan J. Emanuel',
      author_email='alan_emanuel@hms.harvard.edu',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=['numpy','matplotlib','scipy','re','glob','os','skimage',
      'multiprocessing','numba','pims','math','sys','struct','time'])
