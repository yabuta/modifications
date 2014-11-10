/* This file is part of VoltDB.
 * Copyright (C) 2008-2014 VoltDB Inc.
 *
 * This file contains original code and/or modifications of original code.
 * Any modifications made by VoltDB Inc. are licensed under the following
 * terms and conditions:
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
/* Copyright (C) 2008 by H-Store Project
 * Brown University
 * Massachusetts Institute of Technology
 * Yale University
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef GCOMPARISONEXPRESSION_H
#define GCOMPARISONEXPRESSION_H

/*
#include "common/common.h"
#include "common/serializeio.h"
#include "common/valuevector.h"

#include "expressions/abstractexpression.h"
#include "expressions/parametervalueexpression.h"
#include "expressions/constantvalueexpression.h"
#include "expressions/tuplevalueexpression.h"
*/

#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"

#include <string>
#include <iostream>
#include <stdio.h>

namespace voltdb {

class GComparisonExpression{

public:

    GComparisonExpression();

    CUDAH GComparisonExpression(ExpressionType e)
    {
        et = static_cast<int>(e);
    };

    CUDAH bool eval(GNValue NV1, GNValue NV2){
        //printf("comparison\n");

        //printf("ok1 %d\n",et);

        switch(et){
        case (EXPRESSION_TYPE_COMPARE_EQUAL):
            //printf("ok2\n");
            return NV1.op_equals_withoutNull(NV2);
        case (EXPRESSION_TYPE_COMPARE_NOTEQUAL):
            return NV1.op_notEquals_withoutNull(NV2);
        case (EXPRESSION_TYPE_COMPARE_LESSTHAN):
            return NV1.op_lessThan_withoutNull(NV2);
        case (EXPRESSION_TYPE_COMPARE_GREATERTHAN):
            return NV1.op_greaterThan_withoutNull(NV2);
        case (EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO):
            return NV1.op_lessThanOrEqual_withoutNull(NV2);
        case (EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO):
            return NV1.op_greaterThanOrEqual_withoutNull(NV2);
        default:
            //printf("ok3\n");
            return false;
        }
        
    }
    
    CUDAH int getET(){
        return et;
    }

private:

    int et;


};

}
#endif
