#ifndef CABANA_VERSION_HPP
#define CABANA_VERSION_HPP

#include <CabanaCore_config.hpp>

#include <string>

namespace Cabana
{

std::string version();

std::string git_commit_hash();

} // end namespace Cabana

#endif // end CABANA_VERSION_HPP
