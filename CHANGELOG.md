# Change Log

## [0.1.0]()

**New Features**

- Core portable data structures: `Tuple`, `SoA`, `AoSoA`, and `Slice`
- `DeepCopy`: deep copy of data between `AoSoA` data structures
- Sorting and binning `AoSoA` data structures
- `LinkedCellList`: linked cell list implementation
- Portable neighbor list interface
- `VerletList` - linked cell accelerated Verlet list implementation
- Basic tutorial

**Experimental Features (subject to change in future releases)**

- `parallel_for` - portable parallel loops over supported execution spaces
- `neighbor_parallel_for` - portable parallel loops over neighbor lists in supported execution spaces
- `RangePolicy` - defines an index range for parallel loops
