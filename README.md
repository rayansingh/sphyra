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
- `--particles N` or `-p N` - Number of particles (default: 500)

### Examples

```bash
./sphyra
./sphyra --sph=gpu -p 5000 60
./sphyra --sph=gpu --raytracing=gpu -p 10000 30
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
