from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mldp',
    version='0.0.1',
    description='functional machine learning pipeline',
    long_description=readme,
    author='Juniper Overbeck',
    author_email='nodeadtree@gmail.com',
    url='https://github.com/nodeadtree/mldp',
    license=license,
    packages=['mldp/'])
