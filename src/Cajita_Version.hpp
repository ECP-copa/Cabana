#ifndef CAJITA_VERSION_HPP
#define CAJITA_VERSION_HPP

#include <Cajita_config.hpp>

#include <string>

namespace Cajita
{

std::string version();

std::string git_commit_hash();

} // end namespace Cajita

#endif // end CAJITA_VERSION_HPP
