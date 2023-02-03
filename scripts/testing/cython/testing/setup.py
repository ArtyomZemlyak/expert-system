from setuptools import setup
from Cython.Build import cythonize

setup(
    name="hello_world_cy",
    package_dir={"testing": ""},
    ext_modules=cythonize("hello_world_cy.pyx"),
    zip_safe=False,
)
