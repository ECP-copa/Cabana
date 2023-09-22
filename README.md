# Cabana

Cabana is a performance portable library for particle-based simulations.
Applications include, but are not limited to, molecular dynamics (MD) with
short- and/or long-range atomic interactions; various flavors of particle-in-cell
(PIC) methods, including use within fluid/solid mechanics and plasma
physics; N-body cosmology simulations; and peridynamics for fracture mechanics.

Cabana provides particle data structures, algorithms, and communication, as
well as structured grids, grid algorithms, and particle-grid interpolation to
enable simulations on a variety of platforms including many-core CPU and
GPU architectures. Cabana is built on Kokkos, with many additional
optional library dependencies, including MPI for multi-node simulation.

Cabana is developed as part of the Co-Design Center for Particle Applications
(CoPA) within the Exascale Computing Project (ECP) under the U.S. Department
of Energy. CoPA is a multi-institutional project with developers from ORNL,
LANL, SNL, LLNL, PPNL, and ANL.

## Documentation

Instructions for building Cabana on various platforms, an API reference with
tutorial links, and links to the Doxygen can be found in our
[wiki](https://github.com/ECP-copa/Cabana/wiki).

For Cabana-related questions you can open a GitHub issue to interact with the
developers.

## Contributing

We encourage you to contribute to Cabana! Please check the
[guidelines](CONTRIBUTING.md) on how to do so.

## Citing

If you use Cabana in your work, please cite the [JOSS article](CITATION.bib).
Also consider citing the appropriate [release](https://doi.org/10.5281/zenodo.2558368).

## License

Cabana is distributed under an [open source 3-clause BSD license](LICENSE).
