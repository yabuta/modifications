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

#ifndef _EE_COMMON_COMPACTINGSTRINGPOOL_H_
#define _EE_COMMON_COMPACTINGSTRINGPOOL_H_

#include "structures/CompactingPool.h"

#include <cstdlib>
#include <cuda.h>
#include "../cudaheader.h"

extern "C"{
namespace voltdb
{
    class CompactingStringPool
    {
    public:
        CUDAH CompactingStringPool(int32_t elementSize, int32_t elementsPerBuf);

        CUDAH void* malloc();
        CUDAH void free(void* element);
        CUDAH size_t getBytesAllocated() const;

    private:
        CompactingPool m_pool;
    };
}
}

#endif // _EE_COMMON_COMPACTINGSTRINGPOOL_H_
