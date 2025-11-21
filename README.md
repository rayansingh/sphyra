# Sphyra

SPH-based black hole + accretion disk simulation

## Building

```bash
cmake .
make
```

## Usage

```bash
./sphyra [options] [frames]
```

### Options

- `--sph=MODE` - SPH computation: `cpu` or `gpu` (default: cpu)
- `--raytracing=MODE` - Ray tracing: `cpu` or `gpu` (default: cpu)
- `--binning=MODE` - Spatial binning for SPH: `true` or `false` (default: false, requires --sph=gpu)
- `--shared_mem=MODE` - Shared memory for SPH: `true` or `false` (default: false, requires --binning=true)
- `--overlap=MODE` - CUDA stream overlap: `true` or `false` (default: false, requires GPU)
- `--adaptive=MODE` - Adaptive ray quality: `true` or `false` (default: false, GPU raytracing only)
- `--particles N` or `-p N` - Number of particles (default: 500)

### Examples

```bash
./sphyra
./sphyra --sph=gpu -p 5000 60
./sphyra --sph=gpu --raytracing=gpu --overlap=true --adaptive=true -p 10000 30
./sphyra --sph=gpu --binning=true --shared_mem=true --overlap=true --adaptive=true -p 10000 60
```

## Project Structure

```
sphyra/
├── src/           # Source files (CPU implementation)
├── include/       # Header files
├── tests/         # Unit and integration tests
├── benchmarks/    # Performance benchmarks
└── data/          # Output data
```

## Timeline

- **Week 1 (Sept 29)**: Basic particles with gravity
- **Week 2 (Oct 6)**: Neighbor search, density, pressure
- **Week 3 (Oct 13)**: GPU implementation
- **Week 4-5 (Oct 20-Nov 3)**: Rendering
- **Week 6+ (Nov 3-Deadline)**: CPU/GPU pipelining
