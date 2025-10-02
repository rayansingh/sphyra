# Sphyra

SPH-based black hole + accretion disk simulation

## Building

```bash
mkdir build
cd build
cmake ..
make
./sphyra
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
