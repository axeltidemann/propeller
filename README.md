# propeller
Collection of various files for reading/parsing raw data files, as well as experiments on the datasets.

Several python packages are required in order to use the software. These are all listed in the `requirements.txt` file, so in _principle_ it should be as easy as doing

```
pip install -r requirements.txt
```

in order to get all the required packages, however there are some external dependencies you must install.


## Installation instructions on Ubuntu 14.04

You should use OpenBLAS. This is how to download and install it in a virtual environment:

```
git clone git://github.com/xianyi/OpenBLAS
cd OpenBLAS && make FC=gfortran
make PREFIX=$VIRTUAL_ENV install
```

Make sure to `export LD_LIBRARY_PATH=$VIRTUL_ENV/lib` somewhere that makes sense (e.g. in $VIRTUAL_ENV/bin/postactivate). Then you can get numpy and make it use OpenBLAS the following way:

```
git clone https://github.com/numpy/numpy
cd numpy
cp site.cfg.example site.cfg
emacs site.cfg
```

Uncomment these lines:

```
[openblas]
libraries = openblas
library_dirs = /path/to/virtualenv/lib
include_dirs = /path/to/virtualenv/include
runtime_library_dirs = /path/to/virtualenv/lib
```

Note: just writing $VIRTUAL_ENV does not work here, you must specify the entire path. You need cython in order to do the configuration, so do a `pip install cython`. Then do `python setup.py config`. Make sure that everything seems OK with the OpenBLAS install, e.g. something like:

```
FOUND:
  libraries = ['openblas', 'openblas']
  library_dirs = ['/home/tidemann/.virtualenvs/summer9/lib']
  language = c
  define_macros = [('HAVE_CBLAS', None)]
  runtime_library_dirs = ['/home/tidemann/.virtualenvs/summer9/lib']
```

Continue with:

```
python setup.py build
python setup.py install
```

In the repository there is a file that tests the BLAS installation. This is what I see:

```
> OMP_NUM_THREADS=16 python blas_test.py
version: 1.10.0.dev0+808e4c2
maxint:  9223372036854775807

/home/tidemann/.virtualenvs/summer9/local/lib/python2.7/site-packages/numpy/distutils/system_info.py:635: UserWarning: Specified path  is invalid.
  warnings.warn('Specified path %s is invalid.' % d)
  BLAS info:
   * libraries ['openblas', 'openblas']
   * library_dirs ['/home/tidemann/.virtualenvs/summer9/lib']
   * define_macros [('HAVE_CBLAS', None)]
   * language c
   * runtime_library_dirs ['/home/tidemann/.virtualenvs/summer9/lib']

dot: 0.037911 sec
```

Without OpenBLAS, each of these would take roughly 1.21 seconds on my machine, so a speedup of about 30x. Still don't know why I get that "path invalid" warning.

If you're installing on a headless Ubuntu machine using virtualenvwrapper (which you should), save yourself _eons_ of compiling and dependency misery by not trying to install matplotlib via pip. Instead, you should create the virtualenv with the system site packages, download matplotlib using git and install it by hand. This way, all the required windowing just works over SSH. This would be something like the following:

```
mkvirtualenv summer9 --system-site-packages
workon summer9 # For some reason it is not activated directly after system packages are included
git clone git://github.com/matplotlib/matplotlib.git 
cd matplotlib
python setup.py install 
```

If it is any help, my matplotlib version is 1.5.dev1. You should edit the `requirements.txt` file accordingly, of course. An advantage with using --system-site-packages is that you can harness some of the goodies already compiled (e.g. on Ubuntu this helps a lot, see below), and if you subsequently install any local packages, they will override them. In theory. In practice, what I found was a need to edit my `$VIRTUAL_ENV/lib/python2.7/site-packages/easy-install.pth`, since python looks in all *.pth files before loading packages. (I know. Pretty crazy.) Mine had the following:

```
import sys; sys.__plen = len(sys.path)
./matplotlib-1.5.dev1-py2.7-linux-x86_64.egg
/usr/lib/python2.7/dist-packages
import sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:]; p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert = p+len(new)
```

which would load system-wide packages before the local ones. This led to some "mild frustration". I amended this by simply adding the local folder position as the second line, e.g.:

```
import sys; sys.__plen = len(sys.path)
./
./matplotlib-1.5.dev1-py2.7-linux-x86_64.egg
/usr/lib/python2.7/dist-packages
import sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:]; p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert = p+len(new)
```

Now any locally installed packages (e.g. numpy) will be loaded before the system wide ones.

Notice that there are also a ton of dependencies you need to install on Ubuntu, and I wish I had written them down as I went along, but most of them are easily found using a Google search, or just trying to do an `sudo apt-get install --missing-package--`. You also need to install the [HDF5](https://www.hdfgroup.org/HDF5/release/obtainsrc.html) from source, which I prefer to do in a virtualenv as well, since I have experienced that version changes in HDF5 can mess up things. In general, I'd say that most external dependencies should be downloaded and installed into the virtualenv like this:

```
cd --package-name---
./configure --prefix=$VIRTUAL_ENV
make
make install
```

This should be done for the same reasons you do this with python packages - i.e. keep software versions protected. However, sometimes this is simply not possible. On Mac OS X 10.10 it appears easier to do so, but on Ubuntu this will just take up too much of your time (I'm looking at you PyGTK/pycairo/pygobject/gobject-introspection to name just a _very_ few). If you do this installing HDF5, you need to specify where it is before installing tables (you're already losing faith in the `pip install -r requirements.txt`, I can tell).

```
HDF5_DIR=$VIRTUAL_ENV pip install tables
```
