#include <Cabana_Version.hpp>

namespace Cabana
{

std::string version() { return Cabana_VERSION_STRING; }

std::string git_commit_hash() { return Cabana_GIT_COMMIT_HASH; }

} // end namespace Cabana
