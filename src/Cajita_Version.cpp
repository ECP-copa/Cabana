#include <Cajita_Version.hpp>

namespace Cajita
{

std::string version() { return Cajita_VERSION_STRING; }

std::string git_commit_hash() { return Cajita_GIT_COMMIT_HASH; }

} // end namespace Cajita
