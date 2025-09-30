#include <Cabana_Core.hpp>
#include <Cabana_AoSoA.hpp>
#include <Kokkos_Vector.hpp>

#include <Cabana_ParticleInit.hpp>
#include <limits>
#include <cmath>


using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;

enum ParticleFields {
    CellIndex,
    MacroFactor,
    PositionX,
#if VelocityDimensions == 1
    VelocityX
#elif VelocityDimensions == 2
    VelocityPar,
    VelocityPer
#elif VelocityDimensions == 3
    VelocityX,
    VelocityY,
    VelocityZ
#endif
};


// Designate the types that the particles will hold.
using ParticleDataTypes =
Cabana::MemberTypes<
    int,                          // (0) Cell index
    ParticleFloatType,            // (1) macro factor
    ParticleFloatType,            // (2) x-position
#if VelocityDimensions == 1
    ParticleFloatType             // (3) x-velocity
#elif VelocityDimensions == 2
    ParticleFloatType,            // (3) parallel-velocity
    ParticleFloatType             // (4) perp-velocity
#elif VelocityDimensions == 3
    ParticleFloatType,            // (3) x-velocity
    ParticleFloatType,            // (4) y-velocity
    ParticleFloatType             // (5) z-velocity
#endif
>;

// Set the type for the particle AoSoA.
using particle_list_t = Cabana::AoSoA<ParticleDataTypes,MemorySpace>;


// Set the type for a list of Gaussians
using gaussian_list_t = Kokkos::View<GmmFloatType**[n_gaussian_param],Kokkos::HostSpace>;
using gaussian_dev_t  = Kokkos::View<GmmFloatType**[n_gaussian_param],Kokkos::LayoutRight>; // Should we force both host and device view to be LayoutLeft instead?


enum ParticleGenerators {
	RandomSampling,
	CDFInversion,
	HammersleyEqualDensity,
	HammersleyEqualWeight
};

void check_gaussians(const gaussian_list_t& gaussians) {
	int cmax = gaussians.extent(0);
	int kmax = gaussians.extent(1);
	Kokkos::View<GmmFloatType**[VelocityDimensions][VelocityDimensions]> C("C", cmax,kmax);
	Kokkos::View<GmmFloatType**[VelocityDimensions][VelocityDimensions]> B("B", cmax,kmax);

	auto g_dev = Kokkos::create_mirror_view(MemorySpace(), gaussians);
	Kokkos::deep_copy(g_dev, gaussians);

	auto _cholesky = KOKKOS_LAMBDA(const int& j) {
		for(int c = 0; c < cmax; c++) {
			if(g_dev(c,j,Weight)>0.) {
#if VelocityDimensions == 1
				C(c,j,0,0) = g_dev(c,j,Cxx);
				B(c,j,0,0) = g_dev(c,j,Cxx);
#elif VelocityDimensions == 2
				C(c,j,0,0) = g_dev(c,j,Cparpar); C(c,j,0,1) = g_dev(c,j,Cparper);
				C(c,j,1,0) = g_dev(c,j,Cperpar); C(c,j,1,1) = g_dev(c,j,Cperper);
				GmmFloatType Cj[2][2] = {{g_dev(c,j,Cparpar), g_dev(c,j,Cparper)},
				                         {g_dev(c,j,Cperpar), g_dev(c,j,Cperper)}};
				GmmFloatType Bj[2][2];
				Cabana::Impl::Matrix<GmmFloatType,2>::cholesky(Bj,Cj);
				B(c,j,0,0) = Bj[0][0]; B(c,j,0,1) = Bj[0][1];
				B(c,j,1,0) = Bj[1][0]; B(c,j,1,1) = Bj[1][1];
#elif VelocityDimensions == 3
				C(c,j,0,0) = g_dev(c,j,Cxx); C(c,j,0,1) = g_dev(c,j,Cxy); C(c,j,0,2) = g_dev(c,j,Cxz);
				C(c,j,1,0) = g_dev(c,j,Cyx); C(c,j,1,1) = g_dev(c,j,Cyy); C(c,j,1,2) = g_dev(c,j,Cyz);
				C(c,j,2,0) = g_dev(c,j,Czx); C(c,j,2,1) = g_dev(c,j,Czy); C(c,j,2,2) = g_dev(c,j,Czz);
				GmmFloatType Cj[3][3] = {{g_dev(c,j,Cxx), g_dev(c,j,Cxy), g_dev(c,j,Cxz)},
				                         {g_dev(c,j,Cyx), g_dev(c,j,Cyy), g_dev(c,j,Cyz)},
				                         {g_dev(c,j,Czx), g_dev(c,j,Czy), g_dev(c,j,Czz)}};
				GmmFloatType Bj[3][3];
				Cabana::Impl::Matrix2d<GmmFloatType,3>::cholesky(Bj,Cj);
				B(c,j,0,0) = Bj[0][0]; B(c,j,0,1) = Bj[0][1]; B(c,j,0,2) = Bj[0][2];
				B(c,j,1,0) = Bj[1][0]; B(c,j,1,1) = Bj[1][1]; B(c,j,1,2) = Bj[1][2];
				B(c,j,2,0) = Bj[1][0]; B(c,j,2,1) = Bj[2][1]; B(c,j,2,2) = Bj[2][2];
#endif
			}
		}
	};

	Kokkos::parallel_for("Check gaussians", kmax, _cholesky);

	auto B_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);
	auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);

	for(int c = 0; c < cmax; c++) {
		for(int m = 0; m < kmax; m++) {
			if(gaussians(c,m,Weight)>0.) {
#if VelocityDimensions == 1
				if(std::isnan(B_host(c,m,0,0))) {
					printf("Gaussian %d in cell %d has covariance matrix C=(%e), but is that really symmetric and positive definite, because I get a Cholesky decomposition of (%e)\n", m,c, C_host(c,m,0,0), B_host(c,m,0,0));
					exit(1);
				}
#elif VelocityDimensions == 2
				if(std::isnan(B_host(c,m,0,0))||std::isnan(B_host(c,m,0,1)) ||
				   std::isnan(B_host(c,m,1,0))||std::isnan(B_host(c,m,1,1)) ) {
					printf("Gaussian %d in cell %d has covariance matrix C=((%e,%e),(%e,%e)), but is that really symmetric and positive definite, because I get a Cholesky decomposition of ((%e,%e),(%e,%e))\n", m,c, C_host(c,m,0,0),C_host(c,m,0,1), C_host(c,m,1,0),C_host(c,m,1,1), B_host(c,m,0,0),B_host(c,m,0,1), B_host(c,m,1,0),B_host(c,m,1,1));
					exit(1);
				}
#elif VelocityDimensions == 3
				if(std::isnan(B_host(c,m,0,0))||std::isnan(B_host(c,m,0,1))||std::isnan(B_host(c,m,0,2)) ||
				   std::isnan(B_host(c,m,1,0))||std::isnan(B_host(c,m,1,1))||std::isnan(B_host(c,m,1,2)) ||
				   std::isnan(B_host(c,m,2,0))||std::isnan(B_host(c,m,2,1))||std::isnan(B_host(c,m,2,2)) ) {
					printf("Gaussian %d in cell %d has covariance matrix C=((%e,%e,%e),(%e,%e,%e),(%e,%e,%e)), but is that really symmetric and positive definite, because I get a Cholesky decomposition of ((%e,%e,%e),(%e,%e,%e),(%e,%e,%e))\n", m,c, C_host(c,m,0,0),C_host(c,m,0,1),C_host(c,m,0,2), C_host(c,m,1,0),C_host(c,m,1,1),C_host(c,m,1,2), C_host(c,m,2,0),C_host(c,m,2,1),C_host(c,m,2,2), B_host(c,m,0,0),B_host(c,m,0,1),B_host(c,m,0,2), B_host(c,m,1,0),B_host(c,m,1,1),B_host(c,m,1,2), B_host(c,m,2,0),B_host(c,m,2,1),B_host(c,m,2,2));
					exit(1);
				}
#endif
			}
		}
	}
}

void initialize_particles(particle_list_t& particles, const gaussian_list_t& gaussians, const int seed, enum ParticleGenerators samplingtype = RandomSampling) {
	size_t Np;

	auto cell       = Cabana::slice<CellIndex>(particles);
	auto macro      = Cabana::slice<MacroFactor>(particles);
	auto position_x = Cabana::slice<PositionX>(particles);
#if VelocityDimensions == 1
	auto velocity_x = Cabana::slice<VelocityX>(particles);

	if(samplingtype == RandomSampling) {
		Np = Cabana::initializeRandomParticles(cell, macro, position_x, velocity_x, gaussians, seed);
	} else if (samplingtype == CDFInversion) {
		Np = Cabana::initializeParticlesFromCDF(cell, macro, position_x, velocity_x, gaussians, seed);
	} else if (samplingtype == HammersleyEqualDensity) {
		Np = Cabana::initializeEqualDensityParticlesWithHammersley(cell, macro, position_x, velocity_x, gaussians);
	} else if (samplingtype == HammersleyEqualWeight) {
		Np = Cabana::initializeEqualWeightParticlesWithHammersley(cell, macro, position_x, velocity_x, gaussians);
	} else {
		fprintf(stderr, "Unimplemented sampling type %d\n", samplingtype);
		exit(1);
	}

#elif VelocityDimensions == 2
	auto velocity_par = Cabana::slice<VelocityPar>(particles);
	auto velocity_per = Cabana::slice<VelocityPer>(particles);
	if(samplingtype == RandomSampling) {
		Np = Cabana::initializeRandomParticles(cell, macro, position_x, velocity_par, velocity_per, gaussians, seed);
	} else if (samplingtype == HammersleyEqualDensity) {
		Np = Cabana::initializeEqualDensityParticlesWithHammersley(cell, macro, position_x, velocity_par, velocity_per, gaussians);
	} else if (samplingtype == HammersleyEqualWeight) {
		Np = Cabana::initializeEqualWeightParticlesWithHammersley(cell, macro, position_x, velocity_par, velocity_per, gaussians);
	} else {
		fprintf(stderr, "Unimplemented sampling type %d\n", samplingtype);
		exit(1);
	}

#elif VelocityDimensions == 3
	auto velocity_x = Cabana::slice<VelocityX>(particles);
	auto velocity_y = Cabana::slice<VelocityY>(particles);
	auto velocity_z = Cabana::slice<VelocityZ>(particles);

	if(samplingtype == RandomSampling) {
		Np = Cabana::initializeRandomParticles(cell, macro, position_x, velocity_x, velocity_y, velocity_z, gaussians, seed);
	} else if (samplingtype == HammersleyEqualDensity) {
		Np = Cabana::initializeEqualDensityParticlesWithHammersley(cell, macro, position_x, velocity_x, velocity_y, velocity_z, gaussians);
	} else if (samplingtype == HammersleyEqualWeight) {
		Np = Cabana::initializeEqualWeightParticlesWithHammersley(cell, macro, position_x, velocity_x, velocity_y, velocity_z, gaussians);
	} else {
		fprintf(stderr, "Unimplemented sampling type %d\n", samplingtype);
		exit(1);
	}

#endif

	particles.resize(Np);

	GmmFloatType N = 0.;
	auto _sum = KOKKOS_LAMBDA(const int& i, GmmFloatType& lN) {
		lN += macro(i);
	};
	Kokkos::parallel_reduce("Particle Scan", particles.size(), _sum, N);

	printf("We have %f particles with total weight %ld\n", N, Np);
	printf("adjusting particle weights by a factor %f\n", Np/N);

	auto _norm = KOKKOS_LAMBDA(const int s, const int i) {
		macro.access(s,i) *= Np/N;
	};
	Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace> vec_policy(0, Np);
	Cabana::simd_parallel_for(vec_policy, _norm, "norm()");

}

void save_particles(particle_list_t particles, const char* name) {
	// Write out particles
	FILE* fp = fopen(name, "w");
	auto pl_host = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), particles);
	auto      macro_host = Cabana::slice<MacroFactor>(pl_host);
	auto       cell_host = Cabana::slice<CellIndex>(pl_host);
	auto position_x_host = Cabana::slice<PositionX>(pl_host);
#if VelocityDimensions == 1
	auto velocity_x_host = Cabana::slice<VelocityX>(pl_host);
	for(size_t i = 0; i < pl_host.size(); i++) {
		fprintf(fp, "%zd %d %f %f %e\n", i, cell_host(i), macro_host(i), position_x_host(i), velocity_x_host(i));
	}
#elif VelocityDimensions == 2
	auto velocity_par_host = Cabana::slice<VelocityPar>(pl_host);
	auto velocity_per_host = Cabana::slice<VelocityPer>(pl_host);
	for(size_t i = 0; i < pl_host.size(); i++) {
		fprintf(fp, "%zd %d %f %f %e %e\n", i, cell_host(i), macro_host(i), position_x_host(i), velocity_par_host(i), velocity_per_host(i));
	}
#elif VelocityDimensions == 3
	auto velocity_x_host = Cabana::slice<VelocityX>(pl_host);
	auto velocity_y_host = Cabana::slice<VelocityY>(pl_host);
	auto velocity_z_host = Cabana::slice<VelocityZ>(pl_host);
	for(size_t i = 0; i < pl_host.size(); i++) {
		fprintf(fp, "%zd %d %f %f %e %e %e\n", i, cell_host(i), macro_host(i), position_x_host(i), velocity_x_host(i), velocity_y_host(i), velocity_z_host(i));
	}
#endif
	fclose(fp);
}

// solve M.A = rho-rho2 for A
void jacobi(const Kokkos::View<ParticleFloatType**>&M, Kokkos::View<ParticleFloatType*>& A, const Kokkos::View<ParticleFloatType*>& rho, const Kokkos::View<ParticleFloatType*>& rho2) {
	// Square matrix
	assert(M.extent(0) == M.extent(1));
	// And matching sizes
	assert(M.extent(0) == A.extent(0));
	assert(M.extent(0) == rho.extent(0));
	assert(M.extent(0) == rho2.extent(0));

	// Initial guess for dA from diagonal elements
	// M(i,i) * A(i) = rho(i) - rho2(i)
	auto _init = KOKKOS_LAMBDA(const int& i) {
		A(i) = (rho(i) - rho2(i)) / M(i,i);
	};
	Kokkos::parallel_for("guess A", A.extent(0), _init);

	double error = 1.;
	for(size_t k = 0; k < A.extent(0)*A.extent(0); k++) {
		// Perform one jacobi iteration
		Kokkos::View<ParticleFloatType*> Anext("Anext", A.extent(0));
		auto _jacobi = KOKKOS_LAMBDA(const size_t& i) {
			double dot = 0.;
			for(size_t j = 0; j < Anext.extent(0); j++) {
				dot += M(i,j) * A(j);
			}
			Anext(i) = (rho(i)-rho2(i) - dot + M(i,i)*A(i)) / M(i,i);
		};

		Kokkos::parallel_for("update A", A.extent(0), _jacobi);

		// Meassure convergence
		auto _sum = KOKKOS_LAMBDA(const int&i, double& lsum) {
			lsum += (Anext(i)-A(i)) * (Anext(i)-A(i));
		};
		double sum = 0.;
		Kokkos::parallel_reduce("squared difference", A.extent(0), _sum, sum);
		//printf("summed squared differences = %e\n", sum);

		// Overwrite A
		auto _copy = KOKKOS_LAMBDA(const int& i) {
			A(i) = Anext(i);
		};
		Kokkos::parallel_for("overwrite A", A.extent(0), _copy);

		if(sum < error) {
			error = sum;
		} else {
			break;
		}
	}
}

KOKKOS_INLINE_FUNCTION double W_TSC(const double x) {
	if(fabs(x) < 0.5) {
		return 0.75 - fabs(x)*fabs(x);
	} else if (fabs(x) < 1.5) {
		return 0.5 * (1.5-fabs(x)) * (1.5-fabs(x));
	} else {
		return 0.;
	}
}

template<class macro_t, class cell_t, class pos_t, class result_t>
	struct ChargeAdd {

		using V = ParticleFloatType;
		using size_type = size_t;
		using value_type = V[];

		KOKKOS_INLINE_FUNCTION
			void operator()(const int particleindex, value_type rho) const {
				for(int cellindex = cell_(particleindex)-2; cellindex <= cell_(particleindex)+2; cellindex++) {
					// we have a ghost cell on the left which shifts indices of rho relative to cell indices
					const double xg = cellindex;
					rho[cellindex+2] += macro_(particleindex) * W_TSC(posx_(particleindex) - xg);
				}
			}

		KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
			for(size_type i = 0; i < value_count; ++i) {
				dst[i] += src[i];
			}
		}

		KOKKOS_INLINE_FUNCTION void init(value_type dst) const {
			for(size_type i = 0; i < value_count; ++i) {
				dst[i] = 0.;
			}
		}

		using has_final = std::true_type;
		KOKKOS_INLINE_FUNCTION void final(value_type dst) const {
			for(size_type i = 0; i < value_count; ++i) { results_(i) = dst[i]; }
		}

		macro_t macro_;
		cell_t cell_;
		pos_t posx_;
		size_type particle_count;
		size_type value_count; // cell count, but the name is special to kokkos
		result_t results_; // total rho, but the name is special to kokkos
	};

// This version parallelizes over particles like a typical particle-in-cell
// code. It needs some extra doing because the output is an array. For more
// than one spatial dimension this leads to problems.
void chargedensity(const particle_list_t& particles, Kokkos::View<ParticleFloatType*>& rho) {

	const size_t numParticles = particles.size();
	const size_t numCells = rho.extent(0);

	auto macro = Cabana::slice<MacroFactor>(particles);
	auto cell  = Cabana::slice<CellIndex>(particles);
	auto posx  = Cabana::slice<PositionX>(particles);

	using macro_t = typeof(macro);
	using cell_t = typeof(cell);
	using pos_t = typeof(posx);
	using result_t = Kokkos::View<ParticleFloatType*>;

	Kokkos::View<ParticleFloatType*> rhotmp("rhotmp", numCells+4);

	Kokkos::parallel_reduce(numParticles, ChargeAdd<macro_t,cell_t,pos_t,result_t>{macro, cell, posx, numParticles, numCells+4, rhotmp});

	auto rhotmp_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhotmp);
	FILE* fp = fopen("debug_cd1.dat", "w");
	double sum = 0.;
	for(size_t i = 0; i < rhotmp_host.extent(0); i++) {
		fprintf(fp, "%ld %f\n", i, rhotmp_host(i));
		sum += rhotmp_host(i);
	}
	fprintf(fp, "#sum = %f\n", sum);
	fclose(fp);

	// Periodic boundaries on rho
	auto _copy = KOKKOS_LAMBDA(const size_t& i) {
		// copy interior
		rho(i) = rhotmp(i+2);
		// fold ghost cells
		if(i == 0) {
			rho(i) += rhotmp(rho.extent(0)+2);
		}
		if(i == 1) {
			rho(i) += rhotmp(rho.extent(0)+3);
		}
		if(i == rho.extent(0)-2) {
			rho(i) += rhotmp(0);
		}
		if(i == rho.extent(0)-1) {
			rho(i) += rhotmp(1);
		}
	};
	Kokkos::parallel_for("boundary of rho", rho.extent(0), _copy);

	auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
	fp = fopen("debug_cd1b.dat", "w");
	sum = 0.;
	for(size_t i = 0; i < rho_host.extent(0); i++) {
		fprintf(fp, "%ld %f\n", i, rho_host(i));
		sum += rho_host(i);
	}
	fprintf(fp, "#sum = %f\n", sum);
	fclose(fp);
}


// This version parallelizes over cells in rho. It avoids write conflicts but
// processes each particle multiple times which I suspected would be slower.
// But it turns out to run faster in many cases.
void chargedensity2(const particle_list_t& particles, Kokkos::View<ParticleFloatType*>& rho) {

	const size_t numParticles = particles.size();
	const size_t numCells = rho.extent(0);

	auto macro = Cabana::slice<MacroFactor>(particles);
	auto cell  = Cabana::slice<CellIndex>(particles);
	auto posx  = Cabana::slice<PositionX>(particles);

	Kokkos::View<ParticleFloatType*> rhotmp("rhotmp", numCells+4);

	// how to compute a single cell
	auto _deposit = KOKKOS_LAMBDA(const int& cellindex) {
		rhotmp(cellindex) = 0.;
		for(size_t n = 0; n < numParticles; n++) {
			const double xg = cellindex - 2;
			rhotmp[cellindex] += macro(n) * W_TSC(posx(n) - xg);
		}
	};

	Kokkos::parallel_for("deposit rho", rhotmp.extent(0), _deposit);

	auto rhotmp_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhotmp);
	FILE* fp = fopen("debug_cd2.dat", "w");
	double sum = 0.;
	for(size_t i = 0; i < rhotmp_host.extent(0); i++) {
		fprintf(fp, "%ld %f\n", i, rhotmp_host(i));
		sum += rhotmp_host(i);
	}
	fprintf(fp, "#sum = %f\n", sum);
	fclose(fp);

	// Periodic boundaries on rho
	auto _copy = KOKKOS_LAMBDA(const size_t& i) {
		// copy interior
		rho(i) = rhotmp(i+2);
		// fold ghost cells
		if(i == 0) {
			rho(i) += rhotmp(numCells+2);
		}
		if(i == 1) {
			rho(i) += rhotmp(numCells+3);
		}
		if(i == numCells-2) {
			rho(i) += rhotmp(0);
		}
		if(i == numCells-1) {
			rho(i) += rhotmp(1);
		}
	};
	Kokkos::parallel_for("boundary of rho", numCells, _copy);

	auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
	fp = fopen("debug_cd2b.dat", "w");
	sum = 0.;
	fprintf(fp, "\n");
	for(size_t i = 0; i < rho_host.extent(0); i++) {
		fprintf(fp, "%ld %f\n", i, rho_host(i));
		sum += rho_host(i);
	}
	fprintf(fp, "#sum = %f\n", sum);
	fclose(fp);
}


// This version parallelizes over cells in rho. It avoids write conflicts but
// processes each particle multiple times which I suspected would be slower.
// But it turns out to run faster in many cases.
template<typename FunctorType>
void chargedensity3(FunctorType& deposit, const particle_list_t& particles, Kokkos::View<ParticleFloatType*>& rho, Kokkos::View<ParticleFloatType*>& rhotmp) {

	const size_t numParticles = particles.size();
	const size_t numCells = rho.extent(0);

	Kokkos::parallel_for("deposit rho", rhotmp.extent(0), deposit);

	auto rhotmp_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhotmp);
	FILE* fp = fopen("debug_cd3.dat", "w");
	double sum = 0.;
	for(size_t i = 0; i < rhotmp_host.extent(0); i++) {
		fprintf(fp, "%ld %f\n", i, rhotmp_host(i));
		sum += rhotmp_host(i);
	}
	fprintf(fp, "#sum = %f\n", sum);
	fclose(fp);

	// Periodic boundaries on rho
	auto _copy = KOKKOS_LAMBDA(const size_t& i) {
		// copy interior
		rho(i) = rhotmp(i+2);
		// fold ghost cells
		if(i == 0) {
			rho(i) += rhotmp(numCells+2);
		}
		if(i == 1) {
			rho(i) += rhotmp(numCells+3);
		}
		if(i == numCells-2) {
			rho(i) += rhotmp(0);
		}
		if(i == numCells-1) {
			rho(i) += rhotmp(1);
		}
	};
	Kokkos::parallel_for("boundary of rho", numCells, _copy);

	auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
	fp = fopen("debug_cd2b.dat", "w");
	sum = 0.;
	fprintf(fp, "\n");
	for(size_t i = 0; i < rho_host.extent(0); i++) {
		fprintf(fp, "%ld %f\n", i, rho_host(i));
		sum += rho_host(i);
	}
	fprintf(fp, "#sum = %f\n", sum);
	fclose(fp);
}

void massmatrix(const particle_list_t& particles, Kokkos::View<ParticleFloatType**>& M) {

	const size_t numParticles = particles.size();
	const size_t numCells = M.extent(0);

	auto macro = Cabana::slice<MacroFactor>(particles);
	auto cell  = Cabana::slice<CellIndex>(particles);
	auto posx  = Cabana::slice<PositionX>(particles);

	// ghost cells in index i
	Kokkos::View<ParticleFloatType**> Mtmp("Mtmp", numCells+4, M.extent(1));

	// how to compute a single cell
	auto _deposit = KOKKOS_LAMBDA(const int& cellindex) {
		const double xg = cellindex - 2;
		for(size_t n = 0; n < numParticles; n++) {
			size_t j = floor(posx(n)+0.5);
			if(j >= Mtmp.extent(1)) { j = 0; }
			Mtmp(cellindex,j) += macro(n) * W_TSC(posx(n) - xg);
		}
	};

	Kokkos::parallel_for("deposit M", Mtmp.extent(0), _deposit);

	auto _copy = KOKKOS_LAMBDA(const int& j) {
		// copy interior
		for(size_t i = 0; i < numCells; i++) {
			M(i,j) = Mtmp(i+2,j);
		}
		// fold ghost cells
		M(0,j)             += Mtmp(numCells+2,j);
		M(1,j)             += Mtmp(numCells+3,j);
		M(M.extent(0)-2,j) += Mtmp(0,j);
		M(M.extent(0)-1,j) += Mtmp(1,j);
	};
	Kokkos::parallel_for("boundary of M", M.extent(1), _copy);
}


void adjust_for_charge_conservation(particle_list_t& particles, const Kokkos::View<ParticleFloatType*>& dA) {

	auto cell       = Cabana::slice<CellIndex>(particles);
	auto macro      = Cabana::slice<MacroFactor>(particles);
	auto posx       = Cabana::slice<PositionX>(particles);

	auto _update = KOKKOS_LAMBDA(const int s, const int i) {
		int n = (s)*particle_list_t::vector_length + i;
		size_t j = floor(posx(n)+0.5);
		if(j >= dA.extent(0)) { j = 0; }
		ParticleFloatType dalpha_p = dA(j);
		if(macro(n) > -dalpha_p) {
			macro(n) += dalpha_p;
		} else {
			printf("Can't adjust a particle with alpha_p = %f using delta alpha_p = %f in cell %d\n", macro(n), dalpha_p, cell(n));
		}
	};

	// Define an execution policy
	Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace> vec_policy(0, particles.size());

	// Execute for all particles in parallel
	Cabana::simd_parallel_for(vec_policy, _update, "update()");
}


void chargeconservation(const particle_list_t& particles, particle_list_t new_particles, const int num_cells) {
	Kokkos::View<ParticleFloatType*> rho("rho", num_cells); // ghost cells only temporary internally to the function
	//chargedensity(particles, rho);
	chargedensity2(particles, rho);
	auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);

	FILE* fp = fopen("rho.dat", "w");
	for(size_t i = 0; i < rho_host.size(); i++) {
		fprintf(fp, "%zd %f\n", i, rho_host(i));
	}
	fclose(fp);

	Kokkos::View<ParticleFloatType*> rhoprime("rhoprime", num_cells);
	//chargedensity(new_particles, rhoprime);
	chargedensity2(new_particles, rhoprime);
	auto rhoprime_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhoprime);

	fp = fopen("rhoprime.dat", "w");
	for(size_t i = 0; i < rhoprime_host.size(); i++) {
		fprintf(fp, "%zd %f\n", i, rhoprime_host(i));
	}
	fclose(fp);

	Kokkos::View<ParticleFloatType**> M("M", num_cells, num_cells); // ghost cells only temporary internally to the function
	massmatrix(new_particles, M);
	auto M_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), M);

	fp = fopen("M.dat", "w");
	for(size_t i = 0; i < M_host.extent(0); i++) {
		double sum = 0.;
		for(size_t j = 0; j < M_host.extent(1); j++) {
			fprintf(fp, "%zd %zd %f\n", i, j, M_host(i,j));
			sum += M_host(i,j);
		}
		fprintf(fp, "#sum = %f\n", sum);
	}
	fclose(fp);

	Kokkos::View<ParticleFloatType*> dA("dA", num_cells);
	jacobi(M, dA, rho, rhoprime);
	auto dA_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dA);

	fp = fopen("dA.dat", "w");
	for(size_t i = 0; i < dA_host.size(); i++) {
		fprintf(fp, "%zd %f\n", i, dA_host(i));
	}
	fclose(fp);

	adjust_for_charge_conservation(new_particles, dA);

	Kokkos::View<ParticleFloatType*> rhoadj("rhoadj", num_cells);
	//chargedensity(new_particles, rhoadj);
	chargedensity2(new_particles, rhoadj);
	auto rhoadj_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhoadj);

	fp = fopen("rhoadj.dat", "w");
	for(size_t i = 0; i < rhoadj_host.size(); i++) {
		fprintf(fp, "%zd %f\n", i, rhoadj_host(i));
	}
	fclose(fp);
}

void get_stats(const particle_list_t& particles, Kokkos::View<double*>& Np, Kokkos::View<double**>& P, Kokkos::View<double**>& E) {
	auto macro = Cabana::slice<MacroFactor>(particles);
	auto cell  = Cabana::slice<CellIndex>(particles);
#if VelocityDimensions == 1
	auto velX  = Cabana::slice<VelocityX>(particles);
#elif VelocityDimensions == 2
	auto velPar  = Cabana::slice<VelocityPar>(particles);
	auto velPer  = Cabana::slice<VelocityPer>(particles);
#elif VelocityDimensions == 3
	auto velX  = Cabana::slice<VelocityX>(particles);
	auto velY  = Cabana::slice<VelocityY>(particles);
	auto velZ  = Cabana::slice<VelocityZ>(particles);
#endif

	const int numParticles = particles.size();
	const size_t numCells = Np.extent(0);

	auto _moments = KOKKOS_LAMBDA(const int& cellindex) {
		Np(cellindex) = 0.;
		for(int d = 0; d < VelocityDimensions; d++) {
			P(cellindex,d) = 0.;
			E(cellindex,d) = 0.;
		}
		for(int n = 0; n < numParticles; n++) {
			if(cell(n) == cellindex) {
				Np(cellindex) += macro(n);
#if VelocityDimensions == 1
				P(cellindex,0) += macro(n) * velX(n);
				E(cellindex,0) += 0.5*macro(n) * velX(n)*velX(n);
#elif VelocityDimensions == 2
				P(cellindex,0) += macro(n) * velPar(n);
				P(cellindex,1) += macro(n) * velPer(n);
				E(cellindex,0) += 0.5*macro(n) * velPar(n)*velPar(n);
				E(cellindex,1) += 0.5*macro(n) * velPer(n)*velPer(n);
#elif VelocityDimensions == 3
				P(cellindex,0) += macro(n) * velX(n);
				P(cellindex,1) += macro(n) * velY(n);
				P(cellindex,2) += macro(n) * velZ(n);
				E(cellindex,0) += 0.5*macro(n) * velX(n)*velX(n);
				E(cellindex,1) += 0.5*macro(n) * velY(n)*velY(n);
				E(cellindex,2) += 0.5*macro(n) * velZ(n)*velZ(n);
#endif
			}
		}
	};

	Kokkos::parallel_for("compute stats", numCells, _moments);
}

void get_stats(const gaussian_list_t gaussians, const double nppc, Kokkos::View<double**>& P, Kokkos::View<double**>& E) {
	int cmax = gaussians.extent(0);
	int kmax = gaussians.extent(1);
	auto g_dev = Kokkos::create_mirror_view(MemorySpace(), gaussians);
	Kokkos::deep_copy(g_dev, gaussians);

	auto _moments = KOKKOS_LAMBDA(const int& cellindex) {
		for(int d = 0; d < VelocityDimensions; d++) {
			P(cellindex,d) = 0.;
			E(cellindex,d) = 0.;
		}
		for(int k = 0; k < kmax; k++) {
			if(g_dev(cellindex,k,Weight)>0.) {
#if VelocityDimensions == 1
				P(cellindex,0) += nppc*g_dev(cellindex,k,Weight) * g_dev(cellindex,k,MuX);
				E(cellindex,0) += 0.5*nppc * g_dev(cellindex,k,Weight) * (g_dev(cellindex,k,Cxx) + g_dev(cellindex,k,MuX)*g_dev(cellindex,k,MuX));
#elif VelocityDimensions == 2
				P(cellindex,0) += nppc * g_dev(cellindex,k,Weight) * g_dev(cellindex,k,MuPar);
				P(cellindex,1) += nppc * g_dev(cellindex,k,Weight) * Kokkos::sqrt(0.5*M_PI)*Kokkos::sqrt(g_dev(cellindex,k,Cperper));
				E(cellindex,0) += 0.5*nppc * g_dev(cellindex,k,Weight) * (g_dev(cellindex,k,Cparpar) + g_dev(cellindex,k,MuPar)*g_dev(cellindex,k,MuPar));
				E(cellindex,1) += 0.5*nppc * g_dev(cellindex,k,Weight) * (2.*g_dev(cellindex,k,Cperper)                                                 );
#elif VelocityDimensions == 3
				P(cellindex,0) += nppc*g_dev(cellindex,k,Weight)*g_dev(cellindex,k,MuX);
				P(cellindex,1) += nppc*g_dev(cellindex,k,Weight)*g_dev(cellindex,k,MuY);
				P(cellindex,2) += nppc*g_dev(cellindex,k,Weight)*g_dev(cellindex,k,MuZ);
				E(cellindex,0) += 0.5*nppc * g_dev(cellindex,k,Weight) * (g_dev(cellindex,k,Cxx) + g_dev(cellindex,k,MuX)*g_dev(cellindex,k,MuX));
				E(cellindex,1) += 0.5*nppc * g_dev(cellindex,k,Weight) * (g_dev(cellindex,k,Cyy) + g_dev(cellindex,k,MuY)*g_dev(cellindex,k,MuY));
				E(cellindex,2) += 0.5*nppc * g_dev(cellindex,k,Weight) * (g_dev(cellindex,k,Czz) + g_dev(cellindex,k,MuZ)*g_dev(cellindex,k,MuZ));
#endif
			}
		}
	};

	Kokkos::parallel_for("compute stats", cmax, _moments);
}


void get_corrections(const Kokkos::View<double*>& Npprime, const Kokkos::View<double**>& Pprime, const Kokkos::View<double**>& Eprime,
                                                           const Kokkos::View<double**>& P,      const Kokkos::View<double**>& E,
                     Kokkos::View<double**>& a, Kokkos::View<double**>& b) {

	auto _correct = KOKKOS_LAMBDA(const size_t c) {
#if VelocityDimensions == 1
		const double num = 2.*E(c,0)*Npprime(c) - P(c,0)*P(c,0);
		if(num >= 0.) {
			const double denom = 2.*Eprime(c,0)*Npprime(c) - Pprime(c,0)*Pprime(c,0);
			a(c,0) = Kokkos::sqrt(num/denom);
			b(c,0) = (P(c,0)-a(c,0)*Pprime(c,0)) / (a(c,0)*Npprime(c));
		} else{
			a(c,0) = 1.;
			b(c,0) = 0.;
		}
#elif VelocityDimensions == 2
		const double num = 2.*(E(c,0)+E(c,1))*Npprime(c) - 2.*P(c,1)*P(c,1)/Pprime(c,1)/Pprime(c,1)*Eprime(c,1)*Npprime(c) - P(c,0)*P(c,0);
		const double denom = 2.*Eprime(c,0)*Npprime(c) - Pprime(c,0)*Pprime(c,0);
		if(num/denom >= 0.) {
			a(c,0) = Kokkos::sqrt(num/denom);
			a(c,1) = P(c,1) / Pprime(c,1);
			b(c,0) = (P(c,0) - a(c,0)*Pprime(c,0)) / (a(c,0)*Npprime(c));
		} else {
			a(c,0) = 1.;
			a(c,1) = 1.;
			b(c,0) = 0.;
		}
#elif VelocityDimensions == 3
		const double num = 2. * (E(c,0)+E(c,1)+E(c,2)) * Npprime(c) - (P(c,0)*P(c,0) + P(c,1)*P(c,1) + P(c,2)*P(c,2));
		if(num >= 0.) {
			const double denom = 2. * (Eprime(c,0)+Eprime(c,1)+Eprime(c,2)) * Npprime(c) - (Pprime(c,0)*Pprime(c,0) + Pprime(c,1)*Pprime(c,1) + Pprime(c,2)*Pprime(c,2));
			a(c,0) = Kokkos::sqrt(num/denom);
			for(int d = 0; d < VelocityDimensions; d++) {
				b(c,d) = (P(c,d)-a(c,0)*Pprime(c,d)) / (a(c,0)*Npprime(c));
			}
		} else{
			a(c,0) = 1.;
			for(int d = 0; d < VelocityDimensions; d++) {
				b(c,d) = 0.;
			}
		}
#endif
	};

	Kokkos::parallel_for("Compute Lemon corrections", Npprime.extent(0), _correct);
}

void scale_and_shift_particles(particle_list_t& particles, const Kokkos::View<double**>& a, const Kokkos::View<double**>& b) {
	auto cell  = Cabana::slice<CellIndex>(particles);
#if VelocityDimensions == 1
	auto velX  = Cabana::slice<VelocityX>(particles);
#elif VelocityDimensions == 2
	auto velPar  = Cabana::slice<VelocityPar>(particles);
	auto velPer  = Cabana::slice<VelocityPer>(particles);
#elif VelocityDimensions == 3
	auto velX  = Cabana::slice<VelocityX>(particles);
	auto velY  = Cabana::slice<VelocityY>(particles);
	auto velZ  = Cabana::slice<VelocityZ>(particles);
#endif

	// Define how to adjust ONE particle
	auto _adjust = KOKKOS_LAMBDA(const int s, const int i) {
		auto c = cell.access(s,i);
#if VelocityDimensions == 1
		velX.access(s,i) = a(c,0) * (velX.access(s,i) + b(c,0));
#elif VelocityDimensions == 2
		// No shift in vper, different scaling in the two dimensions
		velPar.access(s,i) = a(c,0) * (velPar.access(s,i) + b(c,0));
		velPer.access(s,i) = a(c,1) * (velPer.access(s,i)         );
#elif VelocityDimensions == 3
		// Same scaling for all three dimensions, different shifts in the three dimensions
		velX.access(s,i) = a(c,0) * (velX.access(s,i) + b(c,0));
		velY.access(s,i) = a(c,0) * (velY.access(s,i) + b(c,1));
		velZ.access(s,i) = a(c,0) * (velZ.access(s,i) + b(c,2));
#endif

	};

	// Define an execution policy
	Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace> vec_policy(0, particles.size());

	// Execute
	Cabana::simd_parallel_for(vec_policy, _adjust, "adjust()");

}

//---------------------------------------------------------------------------//
// Main
//---------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
	// Initialize the kokkos runtime.
	Kokkos::ScopeGuard scope_guard( argc, argv );

	const int seed = 12345;
	const int num_cells = 10;
	const int num_particles = 2429*num_cells;

	{
		// Do work here

		// Build outselves a (set of) Gaussian(s)
		gaussian_list_t true_f("true f", num_cells,2); // Ten cells, up to two Gaussians each

#if VelocityDimensions == 1
		// first cell
		true_f(0,0,Weight) = 0.6;
		true_f(0,0,MuX) =  2.0;
		true_f(0,0,Cxx) =  1.5;

		true_f(0,1,Weight) = 0.4;
		true_f(0,1,MuX) = -1.0;
		true_f(0,1,Cxx) =  1.0;

		for(size_t c=1; c<num_cells; c++) {
			true_f(c,0,Weight) = 1.0;
			true_f(c,0,MuX)    = 0.0;
			true_f(c,0,Cxx)    = 1.5;

			true_f(c,1,Weight) = 0.0;
		}
		// Do we even have to set the other parameters for that Gaussian?
#elif VelocityDimensions == 2

		true_f(0,0,Weight) = 0.6;
		true_f(0,0,MuPar)  =  2.0; true_f(0,0,MuPer) =  0.;
		true_f(0,0,Cparpar) = 0.2; true_f(0,0,Cparper) = 0.0;
		true_f(0,0,Cperpar) = 0.0; true_f(0,0,Cperper) = 2.0;

		true_f(0,1,Weight) = 0.4;
		true_f(0,1,MuPar)  =  0.0; true_f(0,1,MuPer) =  1.5;
		true_f(0,1,Cparpar) = 0.05; true_f(0,1,Cparper) = 0.0;
		true_f(0,1,Cperpar) = 0.0; true_f(0,1,Cperper) = 0.1;

		for(size_t c=1; c<num_cells; c++) {
			true_f(c,0,Weight) = 1.0;
			true_f(c,0,MuPar)  =  0.0; true_f(c,0,MuPer) =  0.;
			true_f(c,0,Cparpar) = 0.5; true_f(c,0,Cparper) = 0.0;
			true_f(c,0,Cperpar) = 0.0; true_f(c,0,Cperper) = 0.5;

			true_f(c,1,Weight) = 0.0;
		}
#elif VelocityDimensions == 3
		// first cell
		true_f(0,0,Weight) = 0.6;
		true_f(0,0,MuX) =  2.; true_f(0,0,MuY) =  2.; true_f(0,0,MuZ) =  2.;
		true_f(0,0,Cxx) = 1.5; true_f(0,0,Cxy) = 0.0; true_f(0,0,Cxz) = 0.5;
		true_f(0,0,Cyx) = 0.0; true_f(0,0,Cyy) = 2.0; true_f(0,0,Cyz) = 0.0;
		true_f(0,0,Czx) = 0.5; true_f(0,0,Czy) = 0.0; true_f(0,0,Czz) = 2.5;

		true_f(0,1,Weight) = 0.4;
		true_f(0,1,MuX) = -2.0; true_f(0,1,MuY) = -2.0; true_f(0,1,MuZ) = -2.;
		true_f(0,1,Cxx) = 1.00; true_f(0,1,Cxy) = 0.25; true_f(0,1,Cxz) = 0.0;
		true_f(0,1,Cyx) = 0.25; true_f(0,1,Cyy) = 2.00; true_f(0,1,Cyz) = 1.5;
		true_f(0,1,Czx) = 0.00; true_f(0,1,Czy) = 1.50; true_f(0,1,Czz) = 3.0;

		// second cell
		for(size_t c=1; c<num_cells; c++) {
			true_f(c,0,Weight) = 1.0;
			true_f(c,0,MuX) =  1.; true_f(c,0,MuY) =  2.; true_f(c,0,MuZ) =  3.;
			true_f(c,0,Cxx) = 1.5; true_f(c,0,Cxy) = 0.0; true_f(c,0,Cxz) = 0.0;
			true_f(c,0,Cyx) = 0.0; true_f(c,0,Cyy) = 2.0; true_f(c,0,Cyz) = 0.0;
			true_f(c,0,Czx) = 0.0; true_f(c,0,Czy) = 0.0; true_f(c,0,Czz) = 2.5;

			true_f(c,1,Weight) = 0.0;
		}
		// Do we even have to set the other parameters for that Gaussian?
#endif

		// Make sure we have positive definite covariance matrices
		check_gaussians(true_f);

		FILE* fp = fopen("true_f.dat", "w");
		for(size_t c = 0; c<num_cells; c++) {
			for(size_t m = 0; m < true_f.extent(1); m++) {
				if(true_f(c,m,Weight) > 0.) {
#if VelocityDimensions == 1
					fprintf(fp, "%zd %zd %e %e %e\n", c, m, true_f(c,m,Weight), true_f(c,m,MuX), true_f(c,m,Cxx));
#elif VelocityDimensions == 2
					fprintf(fp, "%zd %zd %e %e %e %e %e %e %e\n", c, m, true_f(c,m,Weight), true_f(c,m,MuPar), true_f(c,m,MuPer), true_f(c,m,Cparpar), true_f(c,m,Cparper), true_f(c,m,Cperpar), true_f(c,m,Cperper));
#elif VelocityDimensions == 3
					fprintf(fp, "%zd %zd %e %e %e %e %e %e %e %e %e %e %e %e %e\n", c, m, true_f(c,m,Weight), true_f(c,m,MuX), true_f(c,m,MuY), true_f(c,m,MuZ), true_f(c,m,Cxx), true_f(c,m,Cxy), true_f(c,m,Cxz), true_f(c,m,Cyx), true_f(c,m,Cyy), true_f(c,m,Cyz), true_f(c,m,Czx), true_f(c,m,Czy), true_f(c,m,Czz));
#endif
				}
			}
		}
		fclose(fp);

		// Create particles
		particle_list_t particles("particles", num_particles);
		printf("Initalize particles\n");
		initialize_particles(particles, true_f, 12345, RandomSampling);

		// Write out particles
		save_particles(particles, "part.dat");

		// Compute GMM for the particles
		size_t kmax = 10;
		const GmmFloatType eps = 1e-16; // eps says what relative decrease in the loss function do we consider CEM converged
		gaussian_list_t gmm("reconstructed f", num_cells, kmax);
		printf("Reconstruct GMM\n");
		//reconstruct_gmm(particles, gmm, eps);

		auto cell       = Cabana::slice<CellIndex>(particles);
		auto macro      = Cabana::slice<MacroFactor>(particles);
#if VelocityDimensions==1
		auto velocity_x = Cabana::slice<VelocityX>(particles);
		Cabana::reconstructGMM(gmm, eps, seed, cell, macro, velocity_x);
#elif VelocityDimensions==2
		auto velocity_par = Cabana::slice<VelocityPar>(particles);
		auto velocity_per = Cabana::slice<VelocityPer>(particles);
		Cabana::reconstructGMM(gmm, eps, seed, cell, macro, velocity_par, velocity_per);
#elif VelocityDimensions==3
		auto velocity_x = Cabana::slice<VelocityX>(particles);
		auto velocity_y = Cabana::slice<VelocityY>(particles);
		auto velocity_z = Cabana::slice<VelocityZ>(particles);
		Cabana::reconstructGMM(gmm, eps, seed, cell, macro, velocity_x, velocity_y, velocity_z);
#endif



		printf("Checking results\n");

		// Take a look at the gaussians
		printf("Reconstructed distribution function:\n");
		for(size_t c = 0; c < gmm.extent(0); c++) {
			for(size_t m = 0; m < kmax; m++) {
				if(gmm(c,m,Weight) > 0.) {
#if VelocityDimensions == 1
					printf("Gaussian(%zd) in cell %zd weight=%f, Mu=%f, Cov=%f\n", m, c, gmm(c,m,Weight), gmm(c,m,MuX), gmm(c,m,Cxx));
#elif VelocityDimensions == 2
					printf("Gaussian(%zd) in cell %zd weight=%f, Mu=(%f,%f), Cov=((%f,%f),(%f,%f))\n", m, c, gmm(c,m,Weight), gmm(c,m,MuPar),gmm(c,m,MuPer),
						gmm(c,m,Cparpar), gmm(c,m,Cparper),
						gmm(c,m,Cperpar), gmm(c,m,Cperper));
#elif VelocityDimensions == 3
					printf("Gaussian(%zd) in cell %zd weight=%f, Mu=(%f,%f,%f), Cov=((%f,%f,%f),(%f,%f,%f),(%f,%f,%f))\n", m, c, gmm(c,m,Weight), gmm(c,m,MuX),gmm(c,m,MuY),gmm(c,m,MuZ),
						gmm(c,m,Cxx), gmm(c,m,Cxy), gmm(c,m,Cxz),
						gmm(c,m,Cyx), gmm(c,m,Cyy), gmm(c,m,Cyz),
						gmm(c,m,Czx), gmm(c,m,Czy), gmm(c,m,Czz));
#endif
				}
			}
		}
		fp = fopen("reconstructed_f.dat", "w");
		for(size_t c = 0; c < gmm.extent(0); c++) {
			for(size_t m = 0; m < gmm.extent(1); m++) {
				if(gmm(c,m,Weight) > 0.) {
#if VelocityDimensions == 1
					fprintf(fp, "%zd %zd %e %e %e\n", c,m, gmm(c,m,Weight), gmm(c,m,MuX), gmm(c,m,Cxx));
#elif VelocityDimensions == 2
					fprintf(fp, "%zd %zd %e %e %e %e %e %e %e\n", c,m, gmm(c,m,Weight), gmm(c,m,MuPar),gmm(c,m,MuPer), gmm(c,m,Cparpar),gmm(c,m,Cparper), gmm(c,m,Cperpar),gmm(c,m,Cperper));
#elif VelocityDimensions == 3
					fprintf(fp, "%zd %zd %e %e %e %e %e %e %e %e %e %e %e %e %e\n", c,m, gmm(c,m,Weight), gmm(c,m,MuX), gmm(c,m,MuY), gmm(c,m,MuZ), gmm(c,m,Cxx), gmm(c,m,Cxy), gmm(c,m,Cxz), gmm(c,m,Cyx), gmm(c,m,Cyy), gmm(c,m,Cyz), gmm(c,m,Czx), gmm(c,m,Czy), gmm(c,m,Czz));
#endif
				}
			}
		}
		fclose(fp);


		// Print ground truth for reference
		printf("True distribution function:\n");
		for(size_t c = 0; c<num_cells; c++) {
			for(size_t m = 0; m < true_f.extent(1); m++) {
				if(true_f(c,m,Weight) > 0.) {
#if VelocityDimensions == 1
					printf("Gaussian(%zd) in cell %zd weight=%f, Mu=%f, Cov=%f\n", m,c, true_f(c,m,Weight), true_f(c,m,MuX), true_f(c,m,Cxx));
#elif VelocityDimensions == 2
					printf("Gaussian(%zd) in cell %zd weight=%f, Mu=(%f,%f), Cov=((%f,%f)(%f,%f))\n", m,c, true_f(c,m,Weight), true_f(c,m,MuPar),true_f(c,m,MuPer),
						true_f(c,m,Cparpar),true_f(c,m,Cparper),
						true_f(c,m,Cperpar),true_f(c,m,Cperper));
#elif VelocityDimensions == 3
					printf("Gaussian(%zd) in cell %zd weight=%f, Mu=(%f,%f,%f), Cov=((%f,%f,%f),(%f,%f,%f),(%f,%f,%f))\n", m,c, true_f(c,m,Weight), true_f(c,m,MuX),true_f(c,m,MuY),true_f(c,m,MuZ),
						true_f(c,m,Cxx), true_f(c,m,Cxy), true_f(c,m,Cxz),
						true_f(c,m,Cyx), true_f(c,m,Cyy), true_f(c,m,Cyz),
						true_f(c,m,Czx), true_f(c,m,Czy), true_f(c,m,Czz));
#endif
				}
			}
		}

		// Create new particles from recovered gaussian distribution functions
		particle_list_t new_particles("new particles", num_particles);
		initialize_particles(new_particles, gmm, 23456, HammersleyEqualWeight);

		// Ensure that the new particles represent the same charge density distribution on cell centers
		chargeconservation(particles, new_particles, num_cells); // this breaks if you use different particle numbers

		// Get statistics from particles
		Kokkos::View<double**> P("P", num_cells, VelocityDimensions);
		Kokkos::View<double**> E("E", num_cells, VelocityDimensions);
		const double nppc = num_particles/(double)num_cells;
		get_stats(gmm, nppc, P, E);

		Kokkos::View<double*> Npprime("Np'", num_cells);
		Kokkos::View<double**> Pprime("P'", num_cells, VelocityDimensions);
		Kokkos::View<double**> Eprime("E'", num_cells, VelocityDimensions);
		get_stats(new_particles, Npprime, Pprime, Eprime);

		Kokkos::View<double**> a("a", num_cells, VelocityDimensions);
		Kokkos::View<double**> b("b", num_cells, VelocityDimensions);
		get_corrections(Npprime, Pprime, Eprime, P, E, a, b);

		auto P_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P);
		auto E_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), E);
		auto Npprime_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Npprime);
		auto Pprime_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Pprime);
		auto Eprime_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Eprime);
		auto a_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
		auto b_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);

		for(size_t c=0; c < num_cells; c++) {
#if VelocityDimensions == 1
			printf("N' = %f, P = %f, P' = %f, E = %f, E' = %f\n", Npprime_host(c), P_host(c,0), Pprime_host(c,0), E_host(c,0), Eprime_host(c,0));
			printf("a = %f, b = %f\n", a_host(c,0), b_host(c,0));
#elif VelocityDimensions == 2
			printf("N' = %f, P = %f,%f, P' = %f,%f, E = %f, E' = %f\n", Npprime_host(c), P_host(c,0),P_host(c,1), Pprime_host(c,0),Pprime_host(c,1), E_host(c,0)+E_host(c,1), Eprime_host(c,0)+Eprime_host(c,1));
			printf("a = %f,%f, b = %f\n", a_host(c,0),a_host(c,1), b_host(c,0));
#elif VelocityDimensions == 3
			printf("N' = %f, P = %f,%f,%f, P' = %f,%f,%f, E = %f, E' = %f\n", Npprime_host(c), P_host(c,0),P_host(c,1),P_host(c,2), Pprime_host(c,0),Pprime_host(c,1),Pprime_host(c,2), E_host(c,0)+E_host(c,1)+E_host(c,2), Eprime_host(c,0)+Eprime_host(c,1)+Eprime_host(c,2));
			printf("a = %f,%f,%f b = %f,%f,%f\n", a_host(c,0),a_host(c,1),a_host(c,2), b_host(c,0),b_host(c,1),b_host(c,2));
#endif
		}

		// Ensure conservation of momentum and energy
		scale_and_shift_particles(new_particles, a, b);

		// Check statistics again
		get_stats(new_particles, Npprime, Pprime, Eprime);

		Kokkos::deep_copy(Npprime_host, Npprime);
		Kokkos::deep_copy(Pprime_host, Pprime);
		Kokkos::deep_copy(Eprime_host, Eprime);

		for(size_t c=0; c < num_cells; c++) {
#if VelocityDimensions == 1
			printf("N' = %f, P = %f, P' = %f, E = %f, E' = %f\n", Npprime_host(c), P_host(c,0), Pprime_host(c,0), E_host(c,0), Eprime_host(c,0));
			printf("deltaP = %f, deltaE = %f\n", P_host(c,0)-Pprime_host(c,0), E_host(c,0)-Eprime_host(c,0));
#elif VelocityDimensions == 2
			printf("N' = %f, P = %f,%f, P' = %f,%f, E = %f, E' = %f\n", Npprime_host(c), P_host(c,0),P_host(c,1), Pprime_host(c,0),Pprime_host(c,1), E_host(c,0)+E_host(c,1), Eprime_host(c,0)+Eprime_host(c,1));
			printf("deltaP = %f,%f, deltaE = %f\n", P_host(c,0)-Pprime_host(c,0), P_host(c,1)-Pprime_host(c,1), E_host(c,0)+E_host(c,1)-Eprime_host(c,0)-Eprime_host(c,1));
#elif VelocityDimensions == 3
			printf("N' = %f, P = %f,%f,%f, P' = %f,%f,%f, E = %f, E' = %f\n", Npprime_host(c), P_host(c,0),P_host(c,1),P_host(c,2), Pprime_host(c,0),Pprime_host(c,1),Pprime_host(c,2), E_host(c,0)+E_host(c,1)+E_host(c,2), Eprime_host(c,0)+Eprime_host(c,1)+Eprime_host(c,2));
			printf("deltaP = %f,%f,%f, deltaE = %f\n", P_host(c,0)-Pprime_host(c,0), P_host(c,1)-Pprime_host(c,1), P_host(c,2)-Pprime_host(c,2), E_host(c,0)+E_host(c,1)+E_host(c,2)-Eprime_host(c,0)-Eprime_host(c,1)-Eprime_host(c,2));
#endif
		}

		// Save particles
		save_particles(new_particles, "part2.dat");
	}

	return 0;
}
