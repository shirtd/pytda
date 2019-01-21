import os, sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True, assume_default_configuration=True,
                        delegate_options_to_subpackages=True, quiet=True)
    config.add_subpackage('pydec')
    config.add_subpackage('tda')
    return config

def setup_package():
    from numpy.distutils.core import setup
    from numpy.distutils.misc_util import Configuration

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)
    sys.path.insert(0, os.path.join(local_path, 'pydec')) # to retrive version
    sys.path.insert(0, os.path.join(local_path, 'tda')) # to retrive version

    try:
        setup(
            name="pytda",
            version="0.0.1",
            author="Kirk Gardner",
            author_email="kirk.gardner@uconn.edu",
            description="Applications of Topological Data Analysis in python.",
            long_description="An exploration of homological sensor networks, "\
                            "circular coordinates, and discrete exterior calculus "\
                            "in python using Dionysus and pydec.",
            url="https://github.com/shirtd/pytda",
            # packages=setuptools.find_packages(),
            configuration=configuration)
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
