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

- `--optimization=LEVEL` - Set optimization level (default: baseline)
- `--particles N` or `-p N` - Number of particles (default: 500)
- `--help` or `-h` - Show help message

### Optimization Levels

The `--optimization` flag is used for performance benchmarking and controls which computational methods are used:

#### `--optimization=baseline`
- Baseline CPU implementation with O(n²) neighbor search

#### `--optimization=gpu_density`
- GPU-accelerated density calculation

### Examples

```bash
# Run with baseline optimization (500 particles, 1 frame)
./sphyra --optimization=baseline

# Run with GPU density optimization (5000 particles, 60 frames)
./sphyra --optimization=gpu_density -p 5000 60

# Run baseline with 10000 particles for 30 frames
./sphyra --optimization=baseline -p 10000 30
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
