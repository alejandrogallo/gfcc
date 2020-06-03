## FAQ

## How to fix the "undefined reference to `hbw_free'" error?

This error might occur during the final stage of the gfcc compilation. Please add the extra option -DTAMM_EXTRA_LIBS="-lmemkind" to cmake.
```
cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH -DTAMM_EXTRA_LIBS="-lmemkind" ..
```

## Job crashes with Bus error

This is a known issue that occurs when limited amount of memory is available.  
https://github.com/spec-org/gfcc/issues/1
