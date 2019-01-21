# import setuptools
#
# setuptools.setup(
#     name="pytda",
#     version="0.0.1",
#     author="Kirk Gardner",
#     author_email="kirk.gardner@uconn.edu",
#     description="Applications of Topological Data Analysis in python.",
#     long_description="An exploration of homological sensor networks, "\
#                     "circular coordinates, and discrete exterior calculus "\
#                     "in python using Dionysus and pydec.",
#     long_description_content_type="text/markdown",
#     url="https://github.com/shirtd/AFRL_TDA",
#     packages=setuptools.find_packages(),
#     install_requires=[
#         "argparse",
#         "cython",
#         "numpy",
#         "scipy",
#         "matplotlib",
#         "dionysus", # requires cmake, boost
#         "scikit-image",
#         "shapely"
#     ]
# )

import os, sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
            assume_default_configuration=True,
            delegate_options_to_subpackages=True,
            quiet=True)

    config.add_subpackage('pydec')
    config.add_subpackage('tda')
    # config.add_data_files(('pydec','*.txt'))

    # config.get_version(os.path.join('pydec','version.py')) # sets config.version

    return config

def setup_package():
    from numpy.distutils.core import setup
    from numpy.distutils.misc_util import Configuration

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)
    sys.path.insert(0,os.path.join(local_path, 'pydec')) # to retrive version
    sys.path.insert(0,os.path.join(local_path, 'tda')) # to retrive version

    try:
        setup(
            name = 'pydec',
            maintainer = "PyDEC Developers",
            maintainer_email = "wnbell@gmail.com",
            # description = DOCLINES[0],
            # long_description = "\n".join(DOCLINES[2:]),
            url = "http://www.graphics.cs.uiuc.edu/~wnbell/",
            download_url = "http://code.google.com/p/pydec/downloads/list",
            license = 'BSD',
            # classifiers=filter(None, CLASSIFIERS.split('\n')),
            platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
            configuration=configuration)
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return

if __name__ == '__main__':
    setup_package()
