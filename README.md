About opt_einsum
================

Home: http://github.com/dgasmith/opt_einsum

Package license: MIT

Feedstock license: BSD 3-Clause

Summary: A contraction optimizer for the NumPy Einsum function.

Einsum is a very powerful function for contracting tensors of arbitrary dimension and index. However, it is only optimized to contract two terms at a time resulting in non-optimal scaling.
For example, let us examine the following index transformation: `M_{pqrs} = C_{pi} C_{qj} I_{ijkl} C_{rk} C_{sl}`

We can then develop two seperate implementations that produce the same result:
```python

N = 10
C = np.random.rand(N, N)
I = np.random.rand(N, N, N, N)

def naive(I, C):
    return np.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)

def optimized(I, C):
    K = np.einsum('pi,ijkl->pjkl', C, I)
    K = np.einsum('qj,pjkl->pqkl', C, K)
    K = np.einsum('rk,pqkl->pqrl', C, K)
    K = np.einsum('sl,pqrl->pqrs', C, K)
    return K
```

The einsum function does not consider building intermediate arrays; therefore, helping einsum out by building these intermediate arrays can result in a considerable cost savings even for small N (N=10):

``` python
np.allclose(naive(I, C), optimized(I, C))
True
%timeit naive(I, C) 1 loops, best of 3: 934 ms per loop
%timeit optimized(I, C) 1000 loops, best of 3: 527 µs per loop
```

A 2000 fold speed up for 4 extra lines of code! This contraction can be further complicated by considering that the shape of the C matrices need not be the same, in this case the ordering in which the indices are transformed matters greatly. Logic can be built that optimizes the ordering; however, this is a lot of time and effort for a single expression. The opt_einsum package is a drop in replacement for the np.einsum function and can handle all of this logic for you:

```
python from opt_einsum import contract
%timeit contract('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
1000 loops, best of 3: 324 µs per loop
```

The above will automatically find the optimal contraction order, in this case identical to that of the optimized function above, and compute the products for you. In this case, it even uses `np.dot` under the hood to exploit any vendor BLAS functionality that your NumPy build has!


Current build status
====================

Linux: [![Circle CI](https://circleci.com/gh/conda-forge/opt_einsum-feedstock.svg?style=shield)](https://circleci.com/gh/conda-forge/opt_einsum-feedstock)
OSX: [![TravisCI](https://travis-ci.org/conda-forge/opt_einsum-feedstock.svg?branch=master)](https://travis-ci.org/conda-forge/opt_einsum-feedstock)
Windows: [![AppVeyor](https://ci.appveyor.com/api/projects/status/github/conda-forge/opt_einsum-feedstock?svg=True)](https://ci.appveyor.com/project/conda-forge/opt-einsum-feedstock/branch/master)

Current release info
====================
Version: [![Anaconda-Server Badge](https://anaconda.org/conda-forge/opt_einsum/badges/version.svg)](https://anaconda.org/conda-forge/opt_einsum)
Downloads: [![Anaconda-Server Badge](https://anaconda.org/conda-forge/opt_einsum/badges/downloads.svg)](https://anaconda.org/conda-forge/opt_einsum)

Installing opt_einsum
=====================

Installing `opt_einsum` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```
conda config --add channels conda-forge
```

Once the `conda-forge` channel has been enabled, `opt_einsum` can be installed with:

```
conda install opt_einsum
```

It is possible to list all of the versions of `opt_einsum` available on your platform with:

```
conda search opt_einsum --channel conda-forge
```


About conda-forge
=================

conda-forge is a community-led conda channel of installable packages.
In order to provide high-quality builds, the process has been automated into the
conda-forge GitHub organization. The conda-forge organization contains one repository
for each of the installable packages. Such a repository is known as a *feedstock*.

A feedstock is made up of a conda recipe (the instructions on what and how to build
the package) and the necessary configurations for automatic building using freely
available continuous integration services. Thanks to the awesome service provided by
[CircleCI](https://circleci.com/), [AppVeyor](http://www.appveyor.com/)
and [TravisCI](https://travis-ci.org/) it is possible to build and upload installable
packages to the [conda-forge](https://anaconda.org/conda-forge)
[Anaconda-Cloud](http://docs.anaconda.org/) channel for Linux, Windows and OSX respectively.

To manage the continuous integration and simplify feedstock maintenance
[conda-smithy](http://github.com/conda-forge/conda-smithy) has been developed.
Using the ``conda-forge.yml`` within this repository, it is possible to re-render all of
this feedstock's supporting files (e.g. the CI configuration files) with ``conda smithy rerender``.


Terminology
===========

**feedstock** - the conda recipe (raw material), supporting scripts and CI configuration.

**conda-smithy** - the tool which helps orchestrate the feedstock.
                   Its primary use is in the construction of the CI ``.yml`` files
                   and simplify the management of *many* feedstocks.

**conda-forge** - the place where the feedstock and smithy live and work to
                  produce the finished article (built conda distributions)


Updating opt_einsum-feedstock
=============================

If you would like to improve the opt_einsum recipe or build a new
package version, please fork this repository and submit a PR. Upon submission,
your changes will be run on the appropriate platforms to give the reviewer an
opportunity to confirm that the changes result in a successful build. Once
merged, the recipe will be re-built and uploaded automatically to the
`conda-forge` channel, whereupon the built conda packages will be available for
everybody to install and use from the `conda-forge` channel.
Note that all branches in the conda-forge/opt_einsum-feedstock are
immediately built and any created packages are uploaded, so PRs should be based
on branches in forks and branches in the main repository should only be used to
build distinct package versions.

In order to produce a uniquely identifiable distribution:
 * If the version of a package **is not** being increased, please add or increase
   the [``build/number``](http://conda.pydata.org/docs/building/meta-yaml.html#build-number-and-string).
 * If the version of a package **is** being increased, please remember to return
   the [``build/number``](http://conda.pydata.org/docs/building/meta-yaml.html#build-number-and-string)
   back to 0.
