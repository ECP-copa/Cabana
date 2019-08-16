/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJTIA_DOMAIN_HPP
#define CAJTIA_DOMAIN_HPP

#include <memory>
#include <vector>

namespace Cajita
{
//---------------------------------------------------------------------------//
/*!
  \class Domain
  \brief Global problem domain.
*/
class Domain
{
  public:
    /*!
     \brief Constructor.
     \param global_low_corner The low corner of the domain in physical space.
     \param global_high_corner The high corner of the domain in physical space.
     \param periodic Whether each logical dimension is periodic.
    */
    Domain( const std::vector<double> &global_low_corner,
            const std::vector<double> &global_high_corner,
            const std::vector<bool> &periodic );

    // Get the global low corner of the domain.
    double lowCorner( const int dim ) const;

    // Get the global high corner of the domain.
    double highCorner( const int dim ) const;

    // Get the extent of a given dimension.
    double extent( const int dim ) const;

    // Get whether a given logical dimension is periodic.
    bool isPeriodic( const int dim ) const;

  private:
    std::vector<double> _global_low_corner;
    std::vector<double> _global_high_corner;
    std::vector<bool> _periodic;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a domain.
  \param global_low_corner The low corner of the domain in physical space.
  \param global_high_corner The high corner of the domain in physical space.
  \param periodic Whether each logical dimension is periodic.
*/
std::shared_ptr<Domain>
createDomain( const std::vector<double> &global_low_corner,
              const std::vector<double> &global_high_corner,
              const std::vector<bool> &periodic );

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJTIA_DOMAIN_HPP
