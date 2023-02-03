from setuptools import setup
from Cython.Build import cythonize

# setup(
#     name='DBESNode',
#     package_dir={'dbes': ''},
#     ext_modules=cythonize("DBESNode.pyx"),
#     zip_safe=False,
# )
setup(
    name="DBESNet",
    package_dir={"dbes": ""},
    ext_modules=cythonize(
        ["DBESNet.pyx", "DBESNode.pyx"], compiler_directives={"boundscheck": False}
    ),
    zip_safe=False,
)
