from setuptools import setup

setup(
     # Needed to silence warnings (and to be a worthwhile package)
    name='jwst_tools',
    url='https://github.com/aggle/jwst_tools',
    author='Jonathan Aguilar',
    author_email='jaguilar@stsci.edu',
    # Needed to actually package something
    packages=['jwst_tools'],
    # Needed for dependencies
    install_requires=['matplotlib',
                      'numpy',
                      'pandas',
                      'jwst',
                      'astropy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Analysis helper tools for JWST data (esp coronagraphic data)',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.md').read(),   
)
