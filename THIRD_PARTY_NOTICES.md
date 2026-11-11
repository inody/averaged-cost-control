# Third-party software notices

This repository's source code is licensed under the MIT License. It depends on third-party packages that remain subject to their own licenses. Their source code is not vendored in this repository; the versions used for the paper reproduction are declared in `pyproject.toml` and `uv.lock`.

## Runtime dependencies

| Package | Version | License |
|---|---:|---|
| NumPy | 2.4.2 | BSD-3-Clause, with additional permissively licensed bundled components |
| SciPy | 1.17.0 | BSD-3-Clause; binary distributions may include separately licensed numerical runtime libraries |
| Matplotlib | 3.10.8 | Matplotlib License (PSF-compatible permissive license) |

## Development and build dependencies

| Package | Version | License |
|---|---:|---|
| pytest | 9.1.1 | MIT |
| Hatchling | version resolved by `uv.lock` | MIT |

Transitive dependencies include software under MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, MIT-CMU, and other permissive licenses. Installed distributions include their applicable license texts. Users redistributing third-party packages or binary environments should retain the notices supplied with those distributions.

This notice is provided for convenience and does not replace the license terms distributed by each third-party project.
