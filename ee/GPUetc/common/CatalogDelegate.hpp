/* This file is part of VoltDB.
 * Copyright (C) 2008-2014 VoltDB Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with VoltDB.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CATALOGDELEGATE_HPP
#define CATALOGDELEGATE_HPP

#include <stdint.h>
#include <string>

#include <cuda.h>
#include "../cudaheader.h"

extern "C" {
namespace voltdb {

/*
 * Interface that bridges catalog textual changes to changes
 * made to actual EE objects. That is, bridge "set /cluster/..
 * tables[A]" to a change to the actual voltdb::Table instance.
 */

class CatalogDelegate {
  public:
  CUDAH CatalogDelegate(int32_t catalogId, std::string path) : m_catalogId(catalogId), m_path(path) {
  }

  CUDAH virtual ~CatalogDelegate() {
  }


  /* Deleted from the catalog */
  CUDAH virtual void deleteCommand() = 0;

  CUDAH void catalogUpdate(int32_t catalogId) {
    m_catalogId = catalogId;
  }

  /* Read the path */
  CUDAH const std::string& path() {
    return m_path;
  }

private:
  /* The catalog id when this delegate was created */
  CUDAH int32_t m_catalogId;

  /* The delegate owns this path and all sub-trees of this path */
  CUDAH const std::string m_path;

};

}
}
#endif
