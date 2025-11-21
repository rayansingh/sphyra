## SPH Parameter Variant Test
If SPH implemented correctly, then:
Changing the SPH constants (SMOOTHING_LENGTH, REST_DENSITY, GAS_CONSTANT, MU) in *constants.h* should show:
- For SMOOTHING_LENGTH:  
  large: stable and clear disk  
  small: randomly floating particles away from the disk

## Frame Comparison Test
- Uncomment the relative lines in *main.cpp*
- Run CUP code and GPU code
- In */test*:
```
make all
make run frame1 frame2
```
