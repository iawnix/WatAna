from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='WatAna',
    version='1.0.0',
    description='XX',
    long_description=readme,
    author='iawnix',
    author_email='iawhaha@163.com',
    url='xx',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

