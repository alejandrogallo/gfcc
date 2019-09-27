## FAQ

## How to fix the "undefined reference to `hbw_free'" error?

cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH -DTAMM_EXTRA_LIBS="-lmemkind" ..

## How to fix the  "Assertion `max_nprim > 0' failed:" error?

A basis set file is missing. This is the command for copying into the installation directories
```
cp  gfcc/contrib/basis/6-31++g*.g94 $GFCC_INSTALL_PATH/share/libint/2.6.0/basis
```
