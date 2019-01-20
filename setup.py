import setuptools

setuptools.setup(
    name="pytda",
    version="0.0.1",
    author="Kirk Gardner",
    author_email="kirk.gardner@uconn.edu",
    description="Applications of Topological Data Analysis in python."
    long_description="An exploration of homological sensor networks, "\
                    "circular coordinates, and discrete exterior calculus "\
                    "in python using Dionysus and pydec.",
    long_description_content_type="text/markdown",
    url="https://github.com/shirtd/AFRL_TDA",
    packages=setuptools.find_packages(),
    install_requires=[
        "argparse",
        "cython",
        "numpy",
        "scipy",
        "matplotlib",
        "dionysus",
        "skimage",
        "shapely"
    ]
)
