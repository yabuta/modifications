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

#ifndef GNVALUE_HPP_
#define GNVALUE_HPP_

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif


#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdint.h>
#include <string>
#include <algorithm>
#include <vector>
#include <stdio.h>

#include "boost/scoped_ptr.hpp"
#include "boost/functional/hash.hpp"
#include "ttmath/ttmathint.h"
#include "common/types.h"
#include "common/value_defs.h"
#include "../cudaheader.h"



namespace voltdb {


/*
 * Objects are length preceded with a short length value or a long length value
 * depending on how many bytes are needed to represent the length. These
 * define how many bytes are used for the short value vs. the long value.
 */
#define SHORT_OBJECT_LENGTHLENGTH static_cast<char>(1)
#define LONG_OBJECT_LENGTHLENGTH static_cast<char>(4)
#define OBJECT_NULL_BIT static_cast<char>(1 << 6)
#define OBJECT_CONTINUATION_BIT static_cast<char>(1 << 7)
#define OBJECT_MAX_LENGTH_SHORT_LENGTH 63

#define FULL_STRING_IN_MESSAGE_THRESHOLD 100

//The int used for storage and return values
typedef ttmath::Int<2> TTInt;
//Long integer with space for multiplication and division without carry/overflow
//typedef ttmath::Int<4> TTLInt;


/**
 * A class to wrap all scalar values regardless of type and
 * storage. An NValue is not the representation used in the
 * serialization of VoltTables nor is it the representation of how
 * scalar values are stored in tables. NValue does have serialization
 * and deserialization mechanisms for both those storage formats.
 * NValues are designed to be immutable and for the most part not
 * constructable from raw data types. Access to the raw data is
 * restricted so that all operations have to go through the member
 * functions that can perform the correct casting and error
 * checking. ValueFactory can be used to construct new NValues, but
 * that should be avoided if possible.
 */
class GNValue {

  public:
    /* Create a default NValue */
    GNValue();

        // todo: free() should not really be const

    /* Release memory associated to object type NValues */
    //void free() const;

    /* Release memory associated to object type tuple columns */
    //static void freeObjectsFromTupleStorage(std::vector<char*> const &oldObjects);

    /* Set value to the correct SQL NULL representation. */
    //void setNull();

    /* Reveal the contained pointer for type values  */
    //oid* castAsAddress() const;

    /* Create a boolean true NValue */
    //static GNValue getTrue();

    /* Create a boolean false NValue */
    //static GNValue getFalse();

    /* Create an NValue with the null representation for valueType */
    //static CUDAH GNValue getNullValue(ValueType);

    /* Create an NValue promoted/demoted to type */
    //GNValue castAs(ValueType type) const;

        // todo: Why doesn't this return size_t? Also, this is a
        // quality of ValueType, not NValue.

    /* Calculate the tuple storage size for an NValue type. VARCHARs
       assume out-of-band tuple storage */
    //static uint16_t getTupleStorageSize(const ValueType type);

       // todo: Could the isInlined argument be removed by have the
       // caller dereference the pointer?

    /* Deserialize a scalar of the specified type from the tuple
       storage area provided. If this is an Object type then the third
       argument indicates whether the object is stored in the tuple
       inline */
    //static CUDAH GNValue initFromTupleStorage(const void *storage, ValueType type, bool isInlined);

    /* Serialize the scalar this NValue represents to the provided
       storage area. If the scalar is an Object type that is not
       inlined then the provided data pool or the heap will be used to
       allocated storage for a copy of the object. */
  /*
    void serializeToTupleStorageAllocateForObjects(
        void *storage, const bool isInlined, const int32_t maxLength,
        const bool isInBytes, Pool *dataPool) const;
  */
    /* Serialize the scalar this NValue represents to the storage area
       provided. If the scalar is an Object type then the object will
       be copy if it can be inlined into the tuple. Otherwise a
       pointer to the object will be copied into the storage area. Any
       allocations needed (if this NValue refers to inlined memory
       whereas the field in the tuple is not inlined), will be done in
       the temp string pool. */
    //CUDAH void serializeToTupleStorage(
  //  void *storage, const bool isInlined, const int32_t maxLength, const bool isInBytes) const;

    /* Deserialize a scalar value of the specified type from the
       SerializeInput directly into the tuple storage area
       provided. This function will perform memory allocations for
       Object types as necessary using the provided data pool or the
       heap. This is used to deserialize tables. */

  /*
    template <TupleSerializationFormat F, Endianess E>
    static void deserializeFrom(
        SerializeInput<E> &input, Pool *dataPool, char *storage,
        const ValueType type, bool isInlined, int32_t maxLength, bool isInBytes);
    static void deserializeFrom(
        SerializeInputBE &input, Pool *dataPool, char *storage,
        const ValueType type, bool isInlined, int32_t maxLength, bool isInBytes);
  */
        // TODO: no callers use the first form; Should combine these
        // eliminate the potential NValue copy.

    /* Read a ValueType from the SerializeInput stream and deserialize
       a scalar value of the specified type into this NValue from the provided
       SerializeInput and perform allocations as necessary. */
    //void deserializeFromAllocateForStorage(SerializeInputBE &input, Pool *dataPool);
    //void deserializeFromAllocateForStorage(ValueType vt, SerializeInputBE &input, Pool *dataPool);

    /* Serialize this NValue to a SerializeOutput */
    //void serializeTo(SerializeOutput &output) const;

    /* Serialize this NValue to an Export stream */
    //void serializeToExport_withoutNull(ExportSerializeOutput&) const;

    // See comment with inlined body, below.  If NULL is supplied for
    // the pool, use the temp string pool.
    //void allocateObjectFromInlinedValue(Pool* pool);

    /* Check if the value represents SQL NULL */
    CUDAH bool isNull() const;

    bool getSourceInlined() const;

    int compare_withoutNull(const GNValue rhs) const;

    /* Return a boolean NValue with the comparison result */

    bool op_equals_withoutNull(const GNValue rhs) const;
    bool op_notEquals_withoutNull(const GNValue rhs) const;
    bool op_lessThan_withoutNull(const GNValue rhs) const;
    bool op_lessThanOrEqual_withoutNull(const GNValue rhs) const;
    bool op_greaterThan_withoutNull(const GNValue rhs) const;
    bool op_greaterThanOrEqual_withoutNull(const GNValue rhs) const;

    static const uint16_t kMaxDecPrec = 38;
    static const uint16_t kMaxDecScale = 12;
    static const int64_t kMaxScaleFactor = 1000000000000;


    CUDAH void setMdata(const char *input){
        memcpy(m_data,input,16);
    }

    CUDAH void setSourceInlined(bool sourceInlined)
    {
        m_sourceInlined = sourceInlined;
    }

    /**
     * Set the type of the value that will be stored in this instance.
     * The last of the 16 bytes of storage allocated in an NValue
     * is used to store the type
     */
    CUDAH void setValueType(ValueType type) {
        m_valueType = type;
    }


  private:


    /**
     * Get the type of the value. This information is private
     * to prevent code outside of NValue from branching based on the type of a value.
     */
    CUDAH ValueType getValueType() const {
        return m_valueType;
    }


    /**
     * 16 bytes of storage for NValue data.
     */
    char m_data[16];
    ValueType m_valueType;
    bool m_sourceInlined;

    /**
     * Private constructor that initializes storage and the specifies the type of value
     * that will be stored in this instance
     */
  CUDAH GNValue(const ValueType type) {
    ::memset( m_data, 0, 16);
    setValueType(type);
    m_sourceInlined = false;
  }


    CUDAH const int8_t& getTinyInt() const {
        assert(getValueType() == VALUE_TYPE_TINYINT);
        return *reinterpret_cast<const int8_t*>(m_data);
    }

    CUDAH int8_t& getTinyInt() {
        assert(getValueType() == VALUE_TYPE_TINYINT);
        return *reinterpret_cast<int8_t*>(m_data);
    }

    CUDAH const int16_t& getSmallInt() const {
        assert(getValueType() == VALUE_TYPE_SMALLINT);
        return *reinterpret_cast<const int16_t*>(m_data);
    }

    CUDAH int16_t& getSmallInt() {
        assert(getValueType() == VALUE_TYPE_SMALLINT);
        return *reinterpret_cast<int16_t*>(m_data);
    }

    CUDAH const int32_t& getInteger() const {
        assert(getValueType() == VALUE_TYPE_INTEGER);
        return *reinterpret_cast<const int32_t*>(m_data);
    }

    CUDAH int32_t& getInteger() {
        assert(getValueType() == VALUE_TYPE_INTEGER);
        return *reinterpret_cast<int32_t*>(m_data);
    }

    CUDAH const int64_t& getBigInt() const {
        assert((getValueType() == VALUE_TYPE_BIGINT) ||
               (getValueType() == VALUE_TYPE_TIMESTAMP) ||
               (getValueType() == VALUE_TYPE_ADDRESS));
        return *reinterpret_cast<const int64_t*>(m_data);
    }

    CUDAH int64_t& getBigInt() {
        assert((getValueType() == VALUE_TYPE_BIGINT) ||
               (getValueType() == VALUE_TYPE_TIMESTAMP) ||
               (getValueType() == VALUE_TYPE_ADDRESS));
        return *reinterpret_cast<int64_t*>(m_data);
    }

    CUDAH const int64_t& getTimestamp() const {
        assert(getValueType() == VALUE_TYPE_TIMESTAMP);
        return *reinterpret_cast<const int64_t*>(m_data);
    }

    CUDAH int64_t& getTimestamp() {
        assert(getValueType() == VALUE_TYPE_TIMESTAMP);
        return *reinterpret_cast<int64_t*>(m_data);
    }

    CUDAH const double& getDouble() const {
        assert(getValueType() == VALUE_TYPE_DOUBLE);
        return *reinterpret_cast<const double*>(m_data);
    }

    CUDAH double& getDouble() {
        assert(getValueType() == VALUE_TYPE_DOUBLE);
        return *reinterpret_cast<double*>(m_data);
    }

    CUDAH const TTInt& getDecimal() const {
        assert(getValueType() == VALUE_TYPE_DECIMAL);
        const void* retval = reinterpret_cast<const void*>(m_data);
        return *reinterpret_cast<const TTInt*>(retval);
    }

    CUDAH TTInt& getDecimal() {
        assert(getValueType() == VALUE_TYPE_DECIMAL);
        void* retval = reinterpret_cast<void*>(m_data);
        return *reinterpret_cast<TTInt*>(retval);
    }

    CUDAH const bool& getBoolean() const {
        assert(getValueType() == VALUE_TYPE_BOOLEAN);
        return *reinterpret_cast<const bool*>(m_data);
    }

    CUDAH bool& getBoolean() {
        assert(getValueType() == VALUE_TYPE_BOOLEAN);
        return *reinterpret_cast<bool*>(m_data);
    }

    bool isBooleanNULL() const ;

  /*
    std::size_t getAllocationSizeForObject() const;
    static std::size_t getAllocationSizeForObject(int32_t length);

    static void throwCastSQLException(const ValueType origType,
                                      const ValueType newType)
    {
        char msg[1024];
        snprintf(msg, 1024, "Type %s can't be cast as %s",
                 valueToString(origType).c_str(),
                 valueToString(newType).c_str());
        throw SQLException(SQLException::
                           data_exception_most_specific_type_mismatch,
                           msg);
    }
  */
    /** return the whole part of a TTInt*/
  /*
    static inline int64_t narrowDecimalToBigInt(TTInt &scaledValue) {
        if (scaledValue > NValue::s_maxInt64AsDecimal || scaledValue < NValue::s_minInt64AsDecimal) {
            throwCastSQLValueOutOfRangeException<TTInt>(scaledValue, VALUE_TYPE_DECIMAL, VALUE_TYPE_BIGINT);
        }
        TTInt whole(scaledValue);
        whole /= kMaxScaleFactor;
        return whole.ToInt();
    }
  */
    /** return the fractional part of a TTInt*/
  /*
    static inline int64_t getFractionalPart(TTInt& scaledValue) {
        TTInt fractional(scaledValue);
        fractional %= kMaxScaleFactor;
        return fractional.ToInt();
    }
  */

    CUDAH int64_t castAsBigIntAndGetValue() const {
        assert(isNull() == false);

        const ValueType type = getValueType();
        assert(type != VALUE_TYPE_NULL);
        switch (type) {
        case VALUE_TYPE_TINYINT:
            return static_cast<int64_t>(getTinyInt());
        case VALUE_TYPE_SMALLINT:
            return static_cast<int64_t>(getSmallInt());
        case VALUE_TYPE_INTEGER:
            return static_cast<int64_t>(getInteger());
        case VALUE_TYPE_BIGINT:
            return getBigInt();
        case VALUE_TYPE_TIMESTAMP:
            return getTimestamp();
        case VALUE_TYPE_DOUBLE:
/*
            if (getDouble() > (double)INT64_MAX || getDouble() < (double)VOLT_INT64_MIN) {
              //throwCastSQLValueOutOfRangeException<double>(getDouble(), VALUE_TYPE_DOUBLE, VALUE_TYPE_BIGINT);
            }
*/
            return static_cast<int64_t>(getDouble());
        case VALUE_TYPE_ADDRESS:
            return getBigInt();
        default:
          //throwCastSQLException(type, VALUE_TYPE_BIGINT);
            return 0; // NOT REACHED
        }
    }

  /*
    int64_t castAsRawInt64AndGetValue() const {
        const ValueType type = getValueType();

        switch (type) {
        case VALUE_TYPE_TINYINT:
            return static_cast<int64_t>(getTinyInt());
        case VALUE_TYPE_SMALLINT:
            return static_cast<int64_t>(getSmallInt());
        case VALUE_TYPE_INTEGER:
            return static_cast<int64_t>(getInteger());
        case VALUE_TYPE_BIGINT:
            return getBigInt();
        case VALUE_TYPE_TIMESTAMP:
            return getTimestamp();
        default:
            throwCastSQLException(type, VALUE_TYPE_BIGINT);
            return 0; // NOT REACHED
        }
    }
  */

  /*
    int32_t castAsIntegerAndGetValue() const {
        assert(isNull() == false);

        const ValueType type = getValueType();
        switch (type) {
        case VALUE_TYPE_NULL:
            return INT32_NULL;
        case VALUE_TYPE_TINYINT:
            return static_cast<int32_t>(getTinyInt());
        case VALUE_TYPE_SMALLINT:
            return static_cast<int32_t>(getSmallInt());
        case VALUE_TYPE_INTEGER:
            return getInteger();
        case VALUE_TYPE_BIGINT:
        {
            const int64_t value = getBigInt();
            if (value > (int64_t)INT32_MAX || value < (int64_t)VOLT_INT32_MIN) {
                throwCastSQLValueOutOfRangeException<int64_t>(value, VALUE_TYPE_BIGINT, VALUE_TYPE_INTEGER);
            }
            return static_cast<int32_t>(value);
        }
        case VALUE_TYPE_DOUBLE:
        {
            const double value = getDouble();
            if (value > (double)INT32_MAX || value < (double)VOLT_INT32_MIN) {
                throwCastSQLValueOutOfRangeException(value, VALUE_TYPE_DOUBLE, VALUE_TYPE_INTEGER);
            }
            return static_cast<int32_t>(value);
        }
        default:
            throwCastSQLException(type, VALUE_TYPE_INTEGER);
            return 0; // NOT REACHED
        }
    }
  */

  /*
    double castAsDoubleAndGetValue() const {
        assert(isNull() == false);

        const ValueType type = getValueType();

        switch (type) {
          case VALUE_TYPE_NULL:
              return DOUBLE_MIN;
          case VALUE_TYPE_TINYINT:
            return static_cast<double>(getTinyInt());
          case VALUE_TYPE_SMALLINT:
            return static_cast<double>(getSmallInt());
          case VALUE_TYPE_INTEGER:
            return static_cast<double>(getInteger());
          case VALUE_TYPE_ADDRESS:
            return static_cast<double>(getBigInt());
          case VALUE_TYPE_BIGINT:
            return static_cast<double>(getBigInt());
          case VALUE_TYPE_TIMESTAMP:
            return static_cast<double>(getTimestamp());
          case VALUE_TYPE_DOUBLE:
            return getDouble();
          case VALUE_TYPE_DECIMAL:
          {
            TTInt scaledValue = getDecimal();
            // we only deal with the decimal number within int64_t range here
            int64_t whole = narrowDecimalToBigInt(scaledValue);
            int64_t fractional = getFractionalPart(scaledValue);
            double retval;
            retval = static_cast<double>(whole) +
                    (static_cast<double>(fractional)/static_cast<double>(kMaxScaleFactor));
            return retval;
          }
          case VALUE_TYPE_VARCHAR:
          case VALUE_TYPE_VARBINARY:
          default:
            throwCastSQLException(type, VALUE_TYPE_DOUBLE);
            return 0; // NOT REACHED
        }
    }
  */
  /*
    TTInt castAsDecimalAndGetValue() const {
        assert(isNull() == false);

        const ValueType type = getValueType();

        switch (type) {
          case VALUE_TYPE_TINYINT:
          case VALUE_TYPE_SMALLINT:
          case VALUE_TYPE_INTEGER:
          case VALUE_TYPE_BIGINT:
          case VALUE_TYPE_TIMESTAMP: {
            int64_t value = castAsRawInt64AndGetValue();
            TTInt retval(value);
            retval *= kMaxScaleFactor;
            return retval;
          }
          case VALUE_TYPE_DECIMAL:
              return getDecimal();
          case VALUE_TYPE_DOUBLE: {
            int64_t intValue = castAsBigIntAndGetValue();
            TTInt retval(intValue);
            retval *= kMaxScaleFactor;

            double value = getDouble();
            value -= static_cast<double>(intValue); // isolate decimal part
            value *= static_cast<double>(kMaxScaleFactor); // scale up to integer.
            TTInt fracval((int64_t)value);
            retval += fracval;
            return retval;
          }
          case VALUE_TYPE_VARCHAR:
          case VALUE_TYPE_VARBINARY:
          default:
            throwCastSQLException(type, VALUE_TYPE_DECIMAL);
            return 0; // NOT REACHED
        }
    }
  */
    /**
     * This funciton does not check NULL value.
     */

  /*
    double getNumberFromString() const
    {
        assert(isNull() == false);

        const int32_t strLength = getObjectLength_withoutNull();
        // Guarantee termination at end of object -- or strtod might not stop there.
        char safeBuffer[strLength+1];
        memcpy(safeBuffer, getObjectValue_withoutNull(), strLength);
        safeBuffer[strLength] = '\0';
        char * bufferEnd = safeBuffer;
        double result = strtod(safeBuffer, &bufferEnd);
        // Needs to have consumed SOMETHING.
        if (bufferEnd > safeBuffer) {
            // Unconsumed trailing chars are OK if they are whitespace.
            while (bufferEnd < safeBuffer+strLength && isspace(*bufferEnd)) {
                ++bufferEnd;
            }
            if (bufferEnd == safeBuffer+strLength) {
                return result;
            }
        }

        std::ostringstream oss;
        oss << "Could not convert to number: '" << safeBuffer << "' contains invalid character value.";
        throw SQLException(SQLException::data_exception_invalid_character_value_for_cast, oss.str().c_str());
    }
  */

  /*
    NValue castAsBigInt() const {
        assert(isNull() == false);

        NValue retval(VALUE_TYPE_BIGINT);
        const ValueType type = getValueType();
        switch (type) {
        case VALUE_TYPE_TINYINT:
            retval.getBigInt() = static_cast<int64_t>(getTinyInt()); break;
        case VALUE_TYPE_SMALLINT:
            retval.getBigInt() = static_cast<int64_t>(getSmallInt()); break;
        case VALUE_TYPE_INTEGER:
            retval.getBigInt() = static_cast<int64_t>(getInteger()); break;
        case VALUE_TYPE_ADDRESS:
            retval.getBigInt() = getBigInt(); break;
        case VALUE_TYPE_BIGINT:
            return *this;
        case VALUE_TYPE_TIMESTAMP:
            retval.getBigInt() = getTimestamp(); break;
        case VALUE_TYPE_DOUBLE:
            if (getDouble() > (double)INT64_MAX || getDouble() < (double)VOLT_INT64_MIN) {
                throwCastSQLValueOutOfRangeException<double>(getDouble(), VALUE_TYPE_DOUBLE, VALUE_TYPE_BIGINT);
            }
            retval.getBigInt() = static_cast<int64_t>(getDouble()); break;
        case VALUE_TYPE_DECIMAL: {
            TTInt scaledValue = getDecimal();
            retval.getBigInt() = narrowDecimalToBigInt(scaledValue); break;
        }
        case VALUE_TYPE_VARCHAR:
            retval.getBigInt() = static_cast<int64_t>(getNumberFromString()); break;
        case VALUE_TYPE_VARBINARY:
        default:
            throwCastSQLException(type, VALUE_TYPE_BIGINT);
        }
        return retval;
    }
  */
  /*
    NValue castAsTimestamp() const {
        assert(isNull() == false);

        NValue retval(VALUE_TYPE_TIMESTAMP);
        const ValueType type = getValueType();
        switch (type) {
        case VALUE_TYPE_TINYINT:
            retval.getTimestamp() = static_cast<int64_t>(getTinyInt()); break;
        case VALUE_TYPE_SMALLINT:
            retval.getTimestamp() = static_cast<int64_t>(getSmallInt()); break;
        case VALUE_TYPE_INTEGER:
            retval.getTimestamp() = static_cast<int64_t>(getInteger()); break;
        case VALUE_TYPE_BIGINT:
            retval.getTimestamp() = getBigInt(); break;
        case VALUE_TYPE_TIMESTAMP:
            retval.getTimestamp() = getTimestamp(); break;
        case VALUE_TYPE_DOUBLE:
            // TODO: Consider just eliminating this switch case to throw a cast exception,
            // or explicitly throwing some other exception here.
            // Direct cast of double to timestamp (implemented via intermediate cast to integer, here)
            // is not a SQL standard requirement, may not even make it past the planner's type-checks,
            // or may just be too far a stretch.
            // OR it might be a convenience for some obscure system-generated edge case?

            if (getDouble() > (double)INT64_MAX || getDouble() < (double)VOLT_INT64_MIN) {
                throwCastSQLValueOutOfRangeException<double>(getDouble(), VALUE_TYPE_DOUBLE, VALUE_TYPE_BIGINT);
            }
            retval.getTimestamp() = static_cast<int64_t>(getDouble()); break;
        case VALUE_TYPE_DECIMAL: {
            // TODO: Consider just eliminating this switch case to throw a cast exception,
            // or explicitly throwing some other exception here.
            // Direct cast of decimal to timestamp (implemented via intermediate cast to integer, here)
            // is not a SQL standard requirement, may not even make it past the planner's type-checks,
            // or may just be too far a stretch.
            // OR it might be a convenience for some obscure system-generated edge case?

            TTInt scaledValue = getDecimal();
            retval.getTimestamp() = narrowDecimalToBigInt(scaledValue); break;
        }
        case VALUE_TYPE_VARCHAR: {
            const int32_t length = getObjectLength_withoutNull();
            const char* bytes = reinterpret_cast<const char*>(getObjectValue_withoutNull());
            const std::string value(bytes, length);
            retval.getTimestamp() = parseTimestampString(value);
            break;
        }
        case VALUE_TYPE_VARBINARY:
        default:
            throwCastSQLException(type, VALUE_TYPE_TIMESTAMP);
        }
        return retval;
    }
  */
  /*
    template <typename T>
    void narrowToInteger(const T value, ValueType sourceType)
    {
        if (value > (T)INT32_MAX || value < (T)VOLT_INT32_MIN) {
            throwCastSQLValueOutOfRangeException(value, sourceType, VALUE_TYPE_INTEGER);
        }
        getInteger() = static_cast<int32_t>(value);
    }
  */

  /*
    NValue castAsInteger() const {
        NValue retval(VALUE_TYPE_INTEGER);
        const ValueType type = getValueType();
        switch (type) {
        case VALUE_TYPE_TINYINT:
            retval.getInteger() = static_cast<int32_t>(getTinyInt()); break;
        case VALUE_TYPE_SMALLINT:
            retval.getInteger() = static_cast<int32_t>(getSmallInt()); break;
        case VALUE_TYPE_INTEGER:
            return *this;
        case VALUE_TYPE_BIGINT:
            retval.narrowToInteger(getBigInt(), type); break;
        case VALUE_TYPE_TIMESTAMP:
            retval.narrowToInteger(getTimestamp(), type); break;
        case VALUE_TYPE_DOUBLE:
            retval.narrowToInteger(getDouble(), type); break;
        case VALUE_TYPE_DECIMAL: {
            TTInt scaledValue = getDecimal();
            // get the whole part of the decimal
            int64_t whole = narrowDecimalToBigInt(scaledValue);
            // try to convert the whole part, which is a int64_t
            retval.narrowToInteger(whole, VALUE_TYPE_BIGINT); break;
        }
        case VALUE_TYPE_VARCHAR:
            retval.narrowToInteger(getNumberFromString(), type); break;
        case VALUE_TYPE_VARBINARY:
        default:
            throwCastSQLException(type, VALUE_TYPE_INTEGER);
        }
        return retval;
    }
  */


  /*
    template <typename T>
    void narrowToSmallInt(const T value, ValueType sourceType)
    {
        if (value > (T)INT16_MAX || value < (T)VOLT_INT16_MIN) {
            throwCastSQLValueOutOfRangeException(value, sourceType, VALUE_TYPE_SMALLINT);
        }
        getSmallInt() = static_cast<int16_t>(value);
    }
  */
  /*
    NValue castAsSmallInt() const {
        assert(isNull() == false);

        NValue retval(VALUE_TYPE_SMALLINT);
        const ValueType type = getValueType();
        switch (type) {
        case VALUE_TYPE_TINYINT:
            retval.getSmallInt() = static_cast<int16_t>(getTinyInt()); break;
        case VALUE_TYPE_SMALLINT:
            retval.getSmallInt() = getSmallInt(); break;
        case VALUE_TYPE_INTEGER:
            retval.narrowToSmallInt(getInteger(), type); break;
        case VALUE_TYPE_BIGINT:
            retval.narrowToSmallInt(getBigInt(), type); break;
        case VALUE_TYPE_TIMESTAMP:
            retval.narrowToSmallInt(getTimestamp(), type); break;
        case VALUE_TYPE_DOUBLE:
            retval.narrowToSmallInt(getDouble(), type); break;
        case VALUE_TYPE_DECIMAL: {
            TTInt scaledValue = getDecimal();
            int64_t whole = narrowDecimalToBigInt(scaledValue);
            retval.narrowToSmallInt(whole, VALUE_TYPE_BIGINT); break;
        }
        case VALUE_TYPE_VARCHAR:
            retval.narrowToSmallInt(getNumberFromString(), type); break;
        case VALUE_TYPE_VARBINARY:
        default:
            throwCastSQLException(type, VALUE_TYPE_SMALLINT);
        }
        return retval;
    }
  */

  /*
    template <typename T>
    void narrowToTinyInt(const T value, ValueType sourceType)
    {
        if (value > (T)INT8_MAX || value < (T)VOLT_INT8_MIN) {
            throwCastSQLValueOutOfRangeException(value, sourceType, VALUE_TYPE_TINYINT);
        }
        getTinyInt() = static_cast<int8_t>(value);
    }
  */
  /*
    NValue castAsTinyInt() const {
        assert(isNull() == false);

        NValue retval(VALUE_TYPE_TINYINT);
        const ValueType type = getValueType();
        switch (type) {
        case VALUE_TYPE_TINYINT:
            retval.getTinyInt() = getTinyInt(); break;
        case VALUE_TYPE_SMALLINT:
            retval.narrowToTinyInt(getSmallInt(), type); break;
        case VALUE_TYPE_INTEGER:
            retval.narrowToTinyInt(getInteger(), type); break;
        case VALUE_TYPE_BIGINT:
            retval.narrowToTinyInt(getBigInt(), type); break;
        case VALUE_TYPE_TIMESTAMP:
            retval.narrowToTinyInt(getTimestamp(), type); break;
        case VALUE_TYPE_DOUBLE:
            retval.narrowToTinyInt(getDouble(), type); break;
        case VALUE_TYPE_DECIMAL: {
            TTInt scaledValue = getDecimal();
            int64_t whole = narrowDecimalToBigInt(scaledValue);
            retval.narrowToTinyInt(whole, type); break;
        }
        case VALUE_TYPE_VARCHAR:
            retval.narrowToTinyInt(getNumberFromString(), type); break;
        case VALUE_TYPE_VARBINARY:
        default:
            throwCastSQLException(type, VALUE_TYPE_TINYINT);
        }
        return retval;
    }
  */

  /*
    NValue castAsDouble() const {
        assert(isNull() == false);

        NValue retval(VALUE_TYPE_DOUBLE);
        const ValueType type = getValueType();
        switch (type) {
        case VALUE_TYPE_TINYINT:
            retval.getDouble() = static_cast<double>(getTinyInt()); break;
        case VALUE_TYPE_SMALLINT:
            retval.getDouble() = static_cast<double>(getSmallInt()); break;
        case VALUE_TYPE_INTEGER:
            retval.getDouble() = static_cast<double>(getInteger()); break;
        case VALUE_TYPE_BIGINT:
            retval.getDouble() = static_cast<double>(getBigInt()); break;
        case VALUE_TYPE_TIMESTAMP:
            retval.getDouble() = static_cast<double>(getTimestamp()); break;
        case VALUE_TYPE_DOUBLE:
            retval.getDouble() = getDouble(); break;
        case VALUE_TYPE_DECIMAL:
            retval.getDouble() = castAsDoubleAndGetValue(); break;
        case VALUE_TYPE_VARCHAR:
            retval.getDouble() = getNumberFromString(); break;
        case VALUE_TYPE_VARBINARY:
        default:
            throwCastSQLException(type, VALUE_TYPE_DOUBLE);
        }
        return retval;
    }
  */

//void streamTimestamp(std::stringstream& value) const;

  /*
    NValue castAsString() const {
        assert(isNull() == false);

        std::stringstream value;
        const ValueType type = getValueType();
        switch (type) {
        case VALUE_TYPE_TINYINT:
            // This cast keeps the tiny int from being confused for a char.
            value << static_cast<int>(getTinyInt()); break;
        case VALUE_TYPE_SMALLINT:
            value << getSmallInt(); break;
        case VALUE_TYPE_INTEGER:
            value << getInteger(); break;
        case VALUE_TYPE_BIGINT:
            value << getBigInt(); break;
        //case VALUE_TYPE_TIMESTAMP:
            //TODO: The SQL standard wants an actual date literal rather than a numeric value, here. See ENG-4284.
            //value << static_cast<double>(getTimestamp()); break;
        case VALUE_TYPE_DOUBLE:
            // Use the specific standard SQL formatting for float values,
            // which the C/C++ format options don't quite support.
            streamSQLFloatFormat(value, getDouble());
            break;
        case VALUE_TYPE_DECIMAL:
            value << createStringFromDecimal(); break;
        case VALUE_TYPE_VARCHAR:
        case VALUE_TYPE_VARBINARY: {
        // note: we allow binary conversion to strings to support
        // byte[] as string parameters...
        // In the future, it would be nice to check this is a decent string here...
            NValue retval(VALUE_TYPE_VARCHAR);
            memcpy(retval.m_data, m_data, sizeof(m_data));
            return retval;
        }
        case VALUE_TYPE_TIMESTAMP: {
            streamTimestamp(value);
            break;
        }
        default:
            throwCastSQLException(type, VALUE_TYPE_VARCHAR);
        }
        return getTempStringValue(value.str().c_str(), value.str().length());
    }
  */
  /*
    NValue castAsBinary() const {
        assert(isNull() == false);

        NValue retval(VALUE_TYPE_VARBINARY);
        const ValueType type = getValueType();
        switch (type) {
        case VALUE_TYPE_VARBINARY:
            memcpy(retval.m_data, m_data, sizeof(m_data));
            break;
        default:
            throwCastSQLException(type, VALUE_TYPE_VARBINARY);
        }
        return retval;
    }
  */

  /*
    void createDecimalFromInt(int64_t rhsint)
    {
        TTInt scaled(rhsint);
        scaled *= kMaxScaleFactor;
        getDecimal() = scaled;
    }
  */

  /*
    static CUDAH inline bool validVarcharSize(const char *valueChars, const size_t length, const int32_t maxLength) {
        int32_t min_continuation_bytes = static_cast<int32_t>(length - maxLength);
        if (min_continuation_bytes <= 0) {
            return true;
        }
        size_t i = length;
        while (i--) {
            if ((valueChars[i] & 0xc0) == 0x80) {
                if (--min_continuation_bytes == 0) {
                    return true;
                }
            }
        }
        return false;
    }
  */
    /**
     * Assuming non-null NValue, validate the size of the varchar or varbinary
     */
  /*
    static inline CUDAH void checkTooNarrowVarcharAndVarbinary(ValueType type, const char* ptr,
            int32_t objLength, int32_t maxLength, bool isInBytes) {
        if (maxLength == 0) {
          //throwFatalLogicErrorStreamed("Zero maxLength for object type " << valueToString(type));
        }

        if (type == VALUE_TYPE_VARBINARY) {
            if (objLength > maxLength) {
                char msg[1024];
                snprintf(msg, 1024,
                        "The size %d of the value exceeds the size of the VARBINARY(%d) column.",
                        objLength, maxLength);
                throw SQLException(SQLException::data_exception_string_data_length_mismatch,
                        msg);
            }
        } else if (type == VALUE_TYPE_VARCHAR) {
            if (isInBytes) {
                if (objLength > maxLength) {
                    std::string inputValue;
                    if (objLength > FULL_STRING_IN_MESSAGE_THRESHOLD) {
                        inputValue = std::string(ptr, FULL_STRING_IN_MESSAGE_THRESHOLD) + std::string("...");
                    } else {
                        inputValue = std::string(ptr, objLength);
                    }
                    char msg[1024];
                    snprintf(msg, 1024,
                            "The size %d of the value '%s' exceeds the size of the VARCHAR(%d BYTES) column.",
                            objLength, inputValue.c_str(), maxLength);
                    throw SQLException(SQLException::data_exception_string_data_length_mismatch,
                            msg);
                }
            } else if (!validVarcharSize(ptr, objLength, maxLength)) {
                const int32_t charLength = getCharLength(ptr, objLength);
                char msg[1024];
                std::string inputValue;
                if (charLength > FULL_STRING_IN_MESSAGE_THRESHOLD) {
                    const char * end = getIthCharPosition(ptr, objLength, FULL_STRING_IN_MESSAGE_THRESHOLD+1);
                    int32_t numBytes = (int32_t)(end - ptr);
                    inputValue = std::string(ptr, numBytes) + std::string("...");
                } else {
                    inputValue = std::string(ptr, objLength);
                }
                snprintf(msg, 1024,
                        "The size %d of the value '%s' exceeds the size of the VARCHAR(%d) column.",
                        charLength, inputValue.c_str(), maxLength);

                throw SQLException(SQLException::data_exception_string_data_length_mismatch,
                        msg);
            }
        } else {
            throwFatalLogicErrorStreamed("NValue::checkTooNarrowVarcharAndVarbinary, "
                    "Invalid object type " << valueToString(type));
        }
    }
  */
    template<typename T>
    CUDAH int compareValue (const T lhsValue, const T rhsValue) const {
        if (lhsValue == rhsValue) {
            return VALUE_COMPARE_EQUAL;
        } else if (lhsValue > rhsValue) {
            return VALUE_COMPARE_GREATERTHAN;
        } else {
            return VALUE_COMPARE_LESSTHAN;
        }
    }

  CUDAH int compareDoubleValue (const double lhsValue, const double rhsValue) const {
        // Treat NaN values as equals and also make them smaller than neagtive infinity.
        // This breaks IEEE754 for expressions slightly.
    /*
        if(std::isnan(lhsValue)){
            return std::isnan(rhsValue) ? VALUE_COMPARE_EQUAL : VALUE_COMPARE_LESSTHAN;
        }
        else if (std::isnan(rhsValue)) {
            return VALUE_COMPARE_GREATERTHAN;
        }
    */

        if (lhsValue > rhsValue) {
            return VALUE_COMPARE_GREATERTHAN;
        }
        else if (lhsValue < rhsValue) {
            return VALUE_COMPARE_LESSTHAN;
        }
        else {
            return VALUE_COMPARE_EQUAL;
        }
    }

    CUDAH int compareTinyInt (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_TINYINT);

        // get the right hand side as a bigint
        if (rhs.getValueType() == VALUE_TYPE_DOUBLE) {
            return compareDoubleValue(static_cast<double>(getTinyInt()), rhs.getDouble());
        } 
        else if (rhs.getValueType() == VALUE_TYPE_DECIMAL) {
            const TTInt rhsValue = rhs.getDecimal();
            TTInt lhsValue(static_cast<int64_t>(getTinyInt()));
            lhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(lhsValue, rhsValue);
        } else {
            int64_t lhsValue, rhsValue;
            lhsValue = static_cast<int64_t>(getTinyInt());
            rhsValue = rhs.castAsBigIntAndGetValue();
            return compareValue<int64_t>(lhsValue, rhsValue);
        }
    }

    CUDAH int compareSmallInt (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_SMALLINT);

        // get the right hand side as a bigint
        if (rhs.getValueType() == VALUE_TYPE_DOUBLE) {
            return compareDoubleValue(static_cast<double>(getSmallInt()), rhs.getDouble());
        } else if (rhs.getValueType() == VALUE_TYPE_DECIMAL) {
            const TTInt rhsValue = rhs.getDecimal();
            TTInt lhsValue(static_cast<int64_t>(getSmallInt()));
            lhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(lhsValue, rhsValue);
        } else {
            int64_t lhsValue, rhsValue;
            lhsValue = static_cast<int64_t>(getSmallInt());
            rhsValue = rhs.castAsBigIntAndGetValue();
            return compareValue<int64_t>(lhsValue, rhsValue);
        }
    }

    CUDAH int compareInteger (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_INTEGER);

        // get the right hand side as a bigint
        if (rhs.getValueType() == VALUE_TYPE_DOUBLE) {
            return compareDoubleValue(static_cast<double>(getInteger()), rhs.getDouble());
        } else if (rhs.getValueType() == VALUE_TYPE_DECIMAL) {
            const TTInt rhsValue = rhs.getDecimal();
            TTInt lhsValue(static_cast<int64_t>(getInteger()));
            lhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(lhsValue, rhsValue);
        } else {
            int64_t lhsValue, rhsValue;
            lhsValue = static_cast<int64_t>(getInteger());
            rhsValue = rhs.castAsBigIntAndGetValue();

            switch (compareValue<int64_t>(lhsValue,rhsValue)) {
            case VALUE_COMPARE_EQUAL:
                return VALUE_COMPARE_EQUAL;
            case VALUE_COMPARE_GREATERTHAN:
                return VALUE_COMPARE_GREATERTHAN;
            case VALUE_COMPARE_LESSTHAN:
                return VALUE_COMPARE_LESSTHAN;
            default:
                return -1;
            }
        }

    }


    CUDAH int compareBigInt (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_BIGINT);

        // get the right hand side as a bigint
        if (rhs.getValueType() == VALUE_TYPE_DOUBLE) {
            return compareDoubleValue(static_cast<double>(getBigInt()), rhs.getDouble());
        } else if (rhs.getValueType() == VALUE_TYPE_DECIMAL) {
            const TTInt rhsValue = rhs.getDecimal();
            TTInt lhsValue(getBigInt());
            lhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(lhsValue, rhsValue);
        } else {
            int64_t lhsValue, rhsValue;
            lhsValue = getBigInt();
            rhsValue = rhs.castAsBigIntAndGetValue();
            return compareValue<int64_t>(lhsValue, rhsValue);
        }
    }

    CUDAH int compareTimestamp (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_TIMESTAMP);

        // get the right hand side as a bigint
        if (rhs.getValueType() == VALUE_TYPE_DOUBLE) {
            return compareDoubleValue(static_cast<double>(getTimestamp()), rhs.getDouble());
        }else if (rhs.getValueType() == VALUE_TYPE_DECIMAL) {
            const TTInt rhsValue = rhs.getDecimal();
            TTInt lhsValue(getTimestamp());
            lhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(lhsValue, rhsValue);
        } else {
            int64_t lhsValue, rhsValue;
            lhsValue = getTimestamp();
            rhsValue = rhs.castAsBigIntAndGetValue();
            return compareValue<int64_t>(lhsValue, rhsValue);
        }
    }

    int compareDoubleValue (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_DOUBLE);

        const double lhsValue = getDouble();
        double rhsValue;

        switch (rhs.getValueType()) {
        case VALUE_TYPE_DOUBLE:
            rhsValue = rhs.getDouble(); break;
        case VALUE_TYPE_TINYINT:
            rhsValue = static_cast<double>(rhs.getTinyInt()); break;
        case VALUE_TYPE_SMALLINT:
            rhsValue = static_cast<double>(rhs.getSmallInt()); break;
        case VALUE_TYPE_INTEGER:
            rhsValue = static_cast<double>(rhs.getInteger()); break;
        case VALUE_TYPE_BIGINT:
            rhsValue = static_cast<double>(rhs.getBigInt()); break;
        case VALUE_TYPE_TIMESTAMP:
            rhsValue = static_cast<double>(rhs.getTimestamp()); break;
        case VALUE_TYPE_DECIMAL:
        {
            TTInt scaledValue = rhs.getDecimal();
            TTInt whole(scaledValue);
            TTInt fractional(scaledValue);
            whole /= kMaxScaleFactor;
            fractional %= kMaxScaleFactor;
            rhsValue = static_cast<double>(whole.ToInt()) +
                    (static_cast<double>(fractional.ToInt())/static_cast<double>(kMaxScaleFactor));
            break;
        }
        default:
/*
            char message[128];
            snprintf(message, 128,
                    "Type %s cannot be cast for comparison to type %s",
                    valueToString(rhs.getValueType()).c_str(),
                    valueToString(getValueType()).c_str());
            throw SQLException(SQLException::
                    data_exception_most_specific_type_mismatch,
                    message);
            // Not reached
            */
            return -3;
        }

        return compareDoubleValue(lhsValue, rhsValue);
    }

/*
    CUDAH int compareStringValue (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_VARCHAR);

        ValueType rhsType = rhs.getValueType();
        if ((rhsType != VALUE_TYPE_VARCHAR) && (rhsType != VALUE_TYPE_VARBINARY)) {
          *
            char message[128];
            snprintf(message, 128,
                    "Type %s cannot be cast for comparison to type %s",
                    valueToString(rhsType).c_str(),
                    valueToString(m_valueType).c_str());
            throw SQLException(SQLException::
                    data_exception_most_specific_type_mismatch,
                    message);
          *
          return -3;
        }

        assert(m_valueType == VALUE_TYPE_VARCHAR);

        const int32_t leftLength = getObjectLength_withoutNull();
        const int32_t rightLength = rhs.getObjectLength_withoutNull();
        const char* left = reinterpret_cast<const char*>(getObjectValue_withoutNull());
        const char* right = reinterpret_cast<const char*>(rhs.getObjectValue_withoutNull());

        const int result = ::strncmp(left, right, std::min(leftLength, rightLength));
        if (result == 0 && leftLength != rightLength) {
            if (leftLength > rightLength) {
                return  VALUE_COMPARE_GREATERTHAN;
            } else {
                return VALUE_COMPARE_LESSTHAN;
            }
        }
        else if (result > 0) {
            return VALUE_COMPARE_GREATERTHAN;
        }
        else if (result < 0) {
            return VALUE_COMPARE_LESSTHAN;
        }

        return VALUE_COMPARE_EQUAL;
    }
*/

/*
    int compareBinaryValue (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_VARBINARY);

        if (rhs.getValueType() != VALUE_TYPE_VARBINARY) {
          *
            char message[128];
            snprintf(message, 128,
                     "Type %s cannot be cast for comparison to type %s",
                     valueToString(rhs.getValueType()).c_str(),
                     valueToString(m_valueType).c_str());
            throw SQLException(SQLException::
                               data_exception_most_specific_type_mismatch,
                               message);
          *
          return -3;
        }
        const int32_t leftLength = getObjectLength_withoutNull();
        const int32_t rightLength = rhs.getObjectLength_withoutNull();

        const char* left = reinterpret_cast<const char*>(getObjectValue_withoutNull());
        const char* right = reinterpret_cast<const char*>(rhs.getObjectValue_withoutNull());

        const int result = ::memcmp(left, right, std::min(leftLength, rightLength));
        if (result == 0 && leftLength != rightLength) {
            if (leftLength > rightLength) {
                return  VALUE_COMPARE_GREATERTHAN;
            } else {
                return VALUE_COMPARE_LESSTHAN;
            }
        }
        else if (result > 0) {
            return VALUE_COMPARE_GREATERTHAN;
        }
        else if (result < 0) {
            return VALUE_COMPARE_LESSTHAN;
        }

        return VALUE_COMPARE_EQUAL;
    }
*/

    int compareDecimalValue (const GNValue rhs) const {

        assert(m_valueType == VALUE_TYPE_DECIMAL);
        switch (rhs.getValueType()) {
        case VALUE_TYPE_DECIMAL:
        {
            return -3;//compareValue<TTInt>(getDecimal(), rhs.getDecimal());
        }
        case VALUE_TYPE_DOUBLE:
        {
            const double rhsValue = rhs.getDouble();
            TTInt scaledValue = getDecimal();
            TTInt whole(scaledValue);
            TTInt fractional(scaledValue);
            whole /= kMaxScaleFactor;
            fractional %= kMaxScaleFactor;
            const double lhsValue = static_cast<double>(whole.ToInt()) +
                    (static_cast<double>(fractional.ToInt())/static_cast<double>(kMaxScaleFactor));

            return compareValue<double>(lhsValue, rhsValue);
        }
        // create the equivalent decimal value
        case VALUE_TYPE_TINYINT:
        {
            TTInt rhsValue(static_cast<int64_t>(rhs.getTinyInt()));
            rhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(getDecimal(), rhsValue);
        }
        case VALUE_TYPE_SMALLINT:
        {
            TTInt rhsValue(static_cast<int64_t>(rhs.getSmallInt()));
            rhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(getDecimal(), rhsValue);
        }
        case VALUE_TYPE_INTEGER:
        {
            TTInt rhsValue(static_cast<int64_t>(rhs.getInteger()));
            rhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(getDecimal(), rhsValue);
        }
        case VALUE_TYPE_BIGINT:
        {
            TTInt rhsValue(rhs.getBigInt());
            rhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(getDecimal(), rhsValue);
        }
        case VALUE_TYPE_TIMESTAMP:
        {
            TTInt rhsValue(rhs.getTimestamp());
            rhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(getDecimal(), rhsValue);
        }
        default:
        {
          /*
            char message[128];
            snprintf(message, 128,
                    "Type %s cannot be cast for comparison to type %s",
                    valueToString(rhs.getValueType()).c_str(),
                    valueToString(getValueType()).c_str());
            throw SQLException(SQLException::
                    data_exception_most_specific_type_mismatch,
                    message);
            // Not reached
            */

        }
        }
    }


  /*
    NValue opAddBigInts(const int64_t lhs, const int64_t rhs) const {
        //Scary overflow check from https://www.securecoding.cert.org/confluence/display/cplusplus/INT32-CPP.+Ensure+that+operations+on+signed+integers+do+not+result+in+overflow
        if ( ((lhs^rhs)
                | (((lhs^(~(lhs^rhs)
                  & (1L << (sizeof(int64_t)*CHAR_BIT-1))))+rhs)^rhs)) >= 0) {
            char message[4096];
            snprintf(message, 4096, "Adding %jd and %jd will overflow BigInt storage", (intmax_t)lhs, (intmax_t)rhs);
            throw SQLException( SQLException::data_exception_numeric_value_out_of_range, message);
        }
        return getBigIntValue(lhs + rhs);
    }
  */
  /*
    NValue opSubtractBigInts(const int64_t lhs, const int64_t rhs) const {
        //Scary overflow check from https://www.securecoding.cert.org/confluence/display/cplusplus/INT32-CPP.+Ensure+that+operations+on+signed+integers+do+not+result+in+overflow
        if ( ((lhs^rhs)
                & (((lhs ^ ((lhs^rhs)
                  & (1L << (sizeof(int64_t)*CHAR_BIT-1))))-rhs)^rhs)) < 0) {
            char message[4096];
            snprintf(message, 4096, "Subtracting %jd from %jd will overflow BigInt storage", (intmax_t)lhs, (intmax_t)rhs);
            throw SQLException( SQLException::data_exception_numeric_value_out_of_range, message);
        }
        return getBigIntValue(lhs - rhs);
    }
  */
  /*
    NValue opMultiplyBigInts(const int64_t lhs, const int64_t rhs) const {
        bool overflow = false;
        //Scary overflow check from https://www.securecoding.cert.org/confluence/display/cplusplus/INT32-CPP.+Ensure+that+operations+on+signed+integers+do+not+result+in+overflow
        if (lhs > 0){  * lhs is positive * 
            if (rhs > 0) {  * lhs and rhs are positive *
                if (lhs > (INT64_MAX / rhs)) {
                    overflow= true;
                }
            } * end if lhs and rhs are positive* 
            else { * lhs positive, rhs non-positive* 
                if (rhs < (INT64_MIN / lhs)) {
                    overflow = true;
                }
            } * lhs positive, rhs non-positive *
        } * end if lhs is positive* 
        else { * lhs is non-positive *
            if (rhs > 0) { * lhs is non-positive, rhs is positive * 
                if (lhs < (INT64_MIN / rhs)) {
                    overflow = true;
                }
            } * end if lhs is non-positive, rhs is positive *
            else { * lhs and rhs are non-positive *
                if ( (lhs != 0) && (rhs < (INT64_MAX / lhs))) {
                    overflow = true;
                }
            } * end if lhs and rhs non-positive *
        } * end if lhs is non-positive *

        const int64_t result = lhs * rhs;

        if (result == INT64_NULL) {
            overflow = true;
        }

        if (overflow) {
            char message[4096];
            snprintf(message, 4096, "Multiplying %jd with %jd will overflow BigInt storage", (intmax_t)lhs, (intmax_t)rhs);
            throw SQLException( SQLException::data_exception_numeric_value_out_of_range, message);
        }

        return getBigIntValue(result);
    }
  */
  
  /*
    NValue opDivideBigInts(const int64_t lhs, const int64_t rhs) const {
        if (rhs == 0) {
            char message[4096];
            snprintf(message, 4096, "Attempted to divide %jd by 0", (intmax_t)lhs);
            throw SQLException(SQLException::data_exception_division_by_zero,
                               message);
        }

        **
         * Because the smallest int64 value is used to represent null (and this is checked for an handled above)
         * it isn't necessary to check for any kind of overflow since none is possible.
         *
        return getBigIntValue(int64_t(lhs / rhs));
    }
  */

  /*
    NValue opAddDoubles(const double lhs, const double rhs) const {
        const double result = lhs + rhs;
        throwDataExceptionIfInfiniteOrNaN(result, "'+' operator");
        return getDoubleValue(result);
    }
  */
  /*
    NValue opSubtractDoubles(const double lhs, const double rhs) const {
        const double result = lhs - rhs;
        throwDataExceptionIfInfiniteOrNaN(result, "'-' operator");
        return getDoubleValue(result);
    }
  */
  /*
    NValue opMultiplyDoubles(const double lhs, const double rhs) const {
        const double result = lhs * rhs;
        throwDataExceptionIfInfiniteOrNaN(result, "'*' operator");
        return getDoubleValue(result);
    }
  */
  /*
    NValue opDivideDoubles(const double lhs, const double rhs) const {
        const double result = lhs / rhs;
        throwDataExceptionIfInfiniteOrNaN(result, "'/' operator");
        return getDoubleValue(result);
    }
  */
  /*
    NValue opAddDecimals(const NValue &lhs, const NValue &rhs) const {
        assert(lhs.isNull() == false);
        assert(rhs.isNull() == false);
        assert(lhs.getValueType() == VALUE_TYPE_DECIMAL);
        assert(rhs.getValueType() == VALUE_TYPE_DECIMAL);

        TTInt retval(lhs.getDecimal());
        if (retval.Add(rhs.getDecimal()) || retval > s_maxDecimalValue || retval < s_minDecimalValue) {
            char message[4096];
            snprintf(message, 4096, "Attempted to add %s with %s causing overflow/underflow",
                    lhs.createStringFromDecimal().c_str(), rhs.createStringFromDecimal().c_str());
            throw SQLException(SQLException::data_exception_numeric_value_out_of_range,
                               message);
        }

        return getDecimalValue(retval);
    }
  */

  /*
    NValue opSubtractDecimals(const NValue &lhs, const NValue &rhs) const {
        assert(lhs.isNull() == false);
        assert(rhs.isNull() == false);
        assert(lhs.getValueType() == VALUE_TYPE_DECIMAL);
        assert(rhs.getValueType() == VALUE_TYPE_DECIMAL);

        TTInt retval(lhs.getDecimal());
        if (retval.Sub(rhs.getDecimal()) || retval > s_maxDecimalValue || retval < s_minDecimalValue) {
            char message[4096];
            snprintf(message, 4096, "Attempted to subtract %s from %s causing overflow/underflow",
                    rhs.createStringFromDecimal().c_str(), lhs.createStringFromDecimal().c_str());
            throw SQLException(SQLException::data_exception_numeric_value_out_of_range,
                               message);
        }

        return getDecimalValue(retval);
    }
  */
    /*
     * Avoid scaling both sides if possible. E.g, don't turn dec * 2 into
     * (dec * 2*kMaxScale*E-12). Then the result of simple multiplication
     * is a*b*E-24 and have to further multiply to get back to the assumed
     * E-12, which can overflow unnecessarily at the middle step.
     */

  /*
    NValue opMultiplyDecimals(const NValue &lhs, const NValue &rhs) const {
        assert(lhs.isNull() == false);
        assert(rhs.isNull() == false);
        assert(lhs.getValueType() == VALUE_TYPE_DECIMAL);
        assert(rhs.getValueType() == VALUE_TYPE_DECIMAL);

        TTLInt calc;
        calc.FromInt(lhs.getDecimal());
        calc *= rhs.getDecimal();
        calc /= kMaxScaleFactor;
        TTInt retval;
        if (retval.FromInt(calc)  || retval > s_maxDecimalValue || retval < s_minDecimalValue) {
            char message[4096];
            snprintf(message, 4096, "Attempted to multiply %s by %s causing overflow/underflow. Unscaled result was %s",
                    lhs.createStringFromDecimal().c_str(), rhs.createStringFromDecimal().c_str(),
                    calc.ToString(10).c_str());
            throw SQLException(SQLException::data_exception_numeric_value_out_of_range,
                               message);
        }
        return getDecimalValue(retval);
    }
  */

    /*
     * Divide two decimals and return a correctly scaled decimal.
     * A little cumbersome. Better algorithms welcome.
     *   (1) calculate the quotient and the remainder.
     *   (2) temporarily scale the remainder to 19 digits
     *   (3) divide out remainder to calculate digits after the radix point.
     *   (4) scale remainder to 12 digits (that's the default scale)
     *   (5) scale the quotient back to 19,12.
     *   (6) sum the scaled quotient and remainder.
     *   (7) construct the final decimal.
     */

  /*
    NValue opDivideDecimals(const NValue &lhs, const NValue &rhs) const {
        assert(lhs.isNull() == false);
        assert(rhs.isNull() == false);
        assert(lhs.getValueType() == VALUE_TYPE_DECIMAL);
        assert(rhs.getValueType() == VALUE_TYPE_DECIMAL);

        TTLInt calc;
        calc.FromInt(lhs.getDecimal());
        calc *= kMaxScaleFactor;
        if (calc.Div(rhs.getDecimal())) {
            char message[4096];
            snprintf( message, 4096, "Attempted to divide %s by %s causing overflow/underflow (or divide by zero)",
                    lhs.createStringFromDecimal().c_str(), rhs.createStringFromDecimal().c_str());
            throw SQLException(SQLException::data_exception_numeric_value_out_of_range,
                               message);
        }
        TTInt retval;
        if (retval.FromInt(calc)  || retval > s_maxDecimalValue || retval < s_minDecimalValue) {
            char message[4096];
            snprintf( message, 4096, "Attempted to divide %s by %s causing overflow. Unscaled result was %s",
                    lhs.createStringFromDecimal().c_str(), rhs.createStringFromDecimal().c_str(),
                    calc.ToString(10).c_str());
            throw SQLException(SQLException::data_exception_numeric_value_out_of_range,
                               message);
        }
        return getDecimalValue(retval);
    }
  */

  /*
    static NValue getTinyIntValue(int8_t value) {
        NValue retval(VALUE_TYPE_TINYINT);
        retval.getTinyInt() = value;
        if (value == INT8_NULL) {
            retval.tagAsNull();
        }
        return retval;
    }
  */

  /*
    static NValue getSmallIntValue(int16_t value) {
        NValue retval(VALUE_TYPE_SMALLINT);
        retval.getSmallInt() = value;
        if (value == INT16_NULL) {
            retval.tagAsNull();
        }
        return retval;
    }
  */

  /*
    static NValue getIntegerValue(int32_t value) {
        NValue retval(VALUE_TYPE_INTEGER);
        retval.getInteger() = value;
        if (value == INT32_NULL) {
            retval.tagAsNull();
        }
        return retval;
    }
  */

  /*
    static NValue getBigIntValue(int64_t value) {
        NValue retval(VALUE_TYPE_BIGINT);
        retval.getBigInt() = value;
        if (value == INT64_NULL) {
            retval.tagAsNull();
        }
        return retval;
    }
  */

  /*
    static NValue getTimestampValue(int64_t value) {
        NValue retval(VALUE_TYPE_TIMESTAMP);
        retval.getTimestamp() = value;
        if (value == INT64_NULL) {
            retval.tagAsNull();
        }
        return retval;
    }
  */

  /*
    static NValue getDoubleValue(double value) {
        NValue retval(VALUE_TYPE_DOUBLE);
        retval.getDouble() = value;
        if (value <= DOUBLE_NULL) {
            retval.tagAsNull();
        }
        return retval;
    }
  */

  /*
    static NValue getDecimalValueFromString(const std::string &value) {
        NValue retval(VALUE_TYPE_DECIMAL);
        retval.createDecimalFromString(value);
        return retval;
    }
  */

  /*
    static NValue getAllocatedArrayValueFromSizeAndType(size_t elementCount, ValueType elementType)
    {
        NValue retval(VALUE_TYPE_ARRAY);
        retval.allocateANewNValueList(elementCount, elementType);
        return retval;
    }
  */
  //static CUDAH Pool* getTempStringPool();

  /*
    static NValue getTempStringValue(const char* value, size_t size) {
        return getAllocatedValue(VALUE_TYPE_VARCHAR, value, size, getTempStringPool());
    }
  */

  /*
    static NValue getTempBinaryValue(const unsigned char* value, size_t size) {
        return getAllocatedValue(VALUE_TYPE_VARBINARY, reinterpret_cast<const char*>(value), size, getTempStringPool());
    }
  */
    /// Assumes hex-encoded input
  /*
    static inline NValue getTempBinaryValueFromHex(const std::string& value) {
        size_t rawLength = value.length() / 2;
        unsigned char rawBuf[rawLength];
        hexDecodeToBinary(rawBuf, value.c_str());
        return getTempBinaryValue(rawBuf, rawLength);
    }
  */

  /*
    static NValue getAllocatedValue(ValueType type, const char* value, size_t size, Pool* stringPool) {
        NValue retval(type);
        char* storage = retval.allocateValueStorage((int32_t)size, stringPool);
        ::memcpy(storage, value, (int32_t)size);
        return retval;
    }
  */

  /*
    char* allocateValueStorage(int32_t length, Pool* stringPool)
    {
        // This unsets the NValue's null tag and returns the length of the length.
        const int8_t lengthLength = setObjectLength(length);
        const int32_t minLength = length + lengthLength;
        StringRef* sref = StringRef::create(minLength, stringPool);
        char* storage = sref->get();
        setObjectLengthToLocation(length, storage);
        storage += lengthLength;
        setObjectValue(sref);
        return storage;
    }
  */

  /*
    static NValue getNullStringValue() {
        NValue retval(VALUE_TYPE_VARCHAR);
        retval.tagAsNull();
        *reinterpret_cast<char**>(retval.m_data) = NULL;
        return retval;
    }
  */

  /*
    static NValue getNullBinaryValue() {
        NValue retval(VALUE_TYPE_VARBINARY);
        retval.tagAsNull();
        *reinterpret_cast<char**>(retval.m_data) = NULL;
        return retval;
    }
  */
    /*
    CUDAH static GNValue getNullValue() {
        GNValue retval(VALUE_TYPE_NULL);
        retval.tagAsNull();
        return retval;
    }
    */

  /*
    static NValue getDecimalValue(TTInt value) {
        NValue retval(VALUE_TYPE_DECIMAL);
        retval.getDecimal() = value;
        return retval;
    }
  */
  /*
    static NValue getAddressValue(void *address) {
        NValue retval(VALUE_TYPE_ADDRESS);
        *reinterpret_cast<void**>(retval.m_data) = address;
        return retval;
    }
  */

};

/**
 * Public constructor that initializes to an NValue that is unusable
 * with other NValues.  Useful for declaring storage for an NValue.
 */
inline CUDAH GNValue::GNValue() {
    ::memset( m_data, 0, 16);
    setValueType(VALUE_TYPE_INVALID);
    m_sourceInlined = false;
}

/**
 * Retrieve a boolean NValue that is true
 */

/*
inline CUDAH GNValue GNValue::getTrue() {
    GNValue retval(VALUE_TYPE_BOOLEAN);
    retval.getBoolean() = true;
    return retval;
}
*/

/**
 * Retrieve a boolean NValue that is false
 */

/*
inline CUDAH GNValue GNValue::getFalse() {
    GNValue retval(VALUE_TYPE_BOOLEAN);
    retval.getBoolean() = false;
    return retval;
}
*/

/**
 * Returns C++ true if this NValue is a boolean and is true
 * If it is NULL, return false.
 */

/*
inline CUDAH bool GNValue::isTrue() const {
    if (isBooleanNULL()) {
        return false;
    }
    return getBoolean();
}
*/

/**
 * Returns C++ false if this NValue is a boolean and is true
 * If it is NULL, return false.
 */

/*
inline bool NValue::isFalse() const {
    if (isBooleanNULL()) {
        return false;
    }
    return !getBoolean();
}

inline bool NValue::isBooleanNULL() const {
    assert(getValueType() == VALUE_TYPE_BOOLEAN);
    return *reinterpret_cast<const int8_t*>(m_data) == INT8_NULL;
}
*/
/*
inline bool NValue::getSourceInlined() const {
    return m_sourceInlined;
}
*/

/**
 * Objects may have storage allocated for them. Calling free causes the NValue to return the storage allocated for
 * the object to the heap
 */

/*
inline void NValue::free() const {
    switch (getValueType())
    {
    case VALUE_TYPE_VARCHAR:
    case VALUE_TYPE_VARBINARY:
    case VALUE_TYPE_ARRAY:
        {
            assert(!m_sourceInlined);
            StringRef* sref = *reinterpret_cast<StringRef* const*>(m_data);
            if (sref != NULL)
            {
                StringRef::destroy(sref);
            }
        }
        break;
    default:
        return;
    }
}

*/

/*
inline void NValue::freeObjectsFromTupleStorage(std::vector<char*> const &oldObjects)
{

    for (std::vector<char*>::const_iterator it = oldObjects.begin(); it != oldObjects.end(); ++it) {
        StringRef* sref = reinterpret_cast<StringRef*>(*it);
        if (sref != NULL) {
            StringRef::destroy(sref);
        }
    }
}
*/

/**
 * Get the amount of storage necessary to store a value of the specified type
 * in a tuple
 */

 /*
inline uint16_t NValue::getTupleStorageSize(const ValueType type) {
    switch (type) {
      case VALUE_TYPE_BIGINT:
      case VALUE_TYPE_TIMESTAMP:
        return sizeof(int64_t);
      case VALUE_TYPE_TINYINT:
        return sizeof(int8_t);
      case VALUE_TYPE_SMALLINT:
        return sizeof(int16_t);
      case VALUE_TYPE_INTEGER:
        return sizeof(int32_t);
      case VALUE_TYPE_DOUBLE:
        return sizeof(double);
      case VALUE_TYPE_VARCHAR:
      case VALUE_TYPE_VARBINARY:
        return sizeof(char*);
      case VALUE_TYPE_DECIMAL:
        return sizeof(TTInt);
      default:
          char message[128];
          snprintf(message, 128, "NValue::getTupleStorageSize() unsupported type '%s'",
                   getTypeName(type).c_str());
          throw SerializableEEException(VOLT_EE_EXCEPTION_TYPE_EEEXCEPTION,
                                        message);
    }
}
 */

/**
 * This null compare function works for GROUP BY, ORDER BY, INDEX KEY, etc,
 * except for comparison expression.
 * comparison expression has different logic for null.
 */

 /*
inline CUDAH int GNValue::compareNull(const GNValue rhs) const {
    bool lnull = isNull();
    bool rnull = rhs.isNull();

    if (lnull) {
        if (rnull) {
            return VALUE_COMPARE_EQUAL;
        } else {
            return VALUE_COMPARE_LESSTHAN;
        }
    } else if (rnull) {
        return VALUE_COMPARE_GREATERTHAN;
    }
    return VALUE_COMPARE_INVALID;
}
 */

/**
 * Assuming no nulls are in comparison.
 * Compare any two NValues. Comparison is not guaranteed to
 * succeed if the values are incompatible.  Avoid use of
 * comparison in favor of op_*.
 */


inline CUDAH int GNValue::compare_withoutNull(const GNValue rhs) const {
    assert(isNull() == false && rhs.isNull() == false);

    switch (m_valueType) {
    case VALUE_TYPE_VARCHAR:
        return 0;//compareStringValue(rhs);
    case VALUE_TYPE_BIGINT:
        return compareBigInt(rhs);
    case VALUE_TYPE_INTEGER:
        return compareInteger(rhs);
    case VALUE_TYPE_SMALLINT:
        return compareSmallInt(rhs);
    case VALUE_TYPE_TINYINT:
        return compareTinyInt(rhs);
    case VALUE_TYPE_TIMESTAMP:
        return compareTimestamp(rhs);
    case VALUE_TYPE_DOUBLE:
        return compareDoubleValue(rhs);
/*
    case VALUE_TYPE_VARBINARY:
    return compareBinaryValue(rhs);
*/
    case VALUE_TYPE_DECIMAL:
        return compareDecimalValue(rhs);
    default: {
        /*
        throwDynamicSQLException(
                "non comparable types lhs '%s' rhs '%s'",
                getValueTypeString().c_str(),
                rhs.getValueTypeString().c_str());
        */
        return 0;
    }
      /* no break */
    }
}


/**
 * Deserialize a scalar value of the specified type from the
 * provided SerializeInput and perform allocations as necessary.
 * This is used to deserialize parameter sets.
 */

/*
inline void NValue::deserializeFromAllocateForStorage(SerializeInputBE &input, Pool *dataPool)
{
    const ValueType type = static_cast<ValueType>(input.readByte());
    deserializeFromAllocateForStorage(type, input, dataPool);
}
*/

 /*
inline void NValue::deserializeFromAllocateForStorage(ValueType type, SerializeInputBE &input, Pool *dataPool)
{
    setValueType(type);
    // Parameter array NValue elements are reused from one executor call to the next,
    // so these NValues need to forget they were ever null.
    m_data[13] = 0; // effectively, this is tagAsNonNull()
    switch (type) {
    case VALUE_TYPE_BIGINT:
        getBigInt() = input.readLong();
        if (getBigInt() == INT64_NULL) {
            tagAsNull();
        }
        break;
    case VALUE_TYPE_TIMESTAMP:
        getTimestamp() = input.readLong();
        if (getTimestamp() == INT64_NULL) {
            tagAsNull();
        }
        break;
    case VALUE_TYPE_TINYINT:
        getTinyInt() = input.readByte();
        if (getTinyInt() == INT8_NULL) {
            tagAsNull();
        }
        break;
    case VALUE_TYPE_SMALLINT:
        getSmallInt() = input.readShort();
        if (getSmallInt() == INT16_NULL) {
            tagAsNull();
        }
        break;
    case VALUE_TYPE_INTEGER:
        getInteger() = input.readInt();
        if (getInteger() == INT32_NULL) {
            tagAsNull();
        }
        break;
    case VALUE_TYPE_DOUBLE:
        getDouble() = input.readDouble();
        if (getDouble() <= DOUBLE_NULL) {
            tagAsNull();
        }
        break;
    case VALUE_TYPE_VARCHAR:
    case VALUE_TYPE_VARBINARY:
    {
        const int32_t length = input.readInt();
        // the NULL SQL string is a NULL C pointer
        if (length == OBJECTLENGTH_NULL) {
            setNull();
            break;
        }
        char* storage = allocateValueStorage(length, dataPool);
        const char *str = (const char*) input.getRawPointer(length);
        ::memcpy(storage, str, length);
        break;
    }
    case VALUE_TYPE_DECIMAL: {
        getDecimal().table[1] = input.readLong();
        getDecimal().table[0] = input.readLong();
        break;
    }
    case VALUE_TYPE_NULL: {
        setNull();
        break;
    }
    case VALUE_TYPE_ARRAY: {
        deserializeIntoANewNValueList(input, dataPool);
        break;
    }
    default:
        throwDynamicSQLException("NValue::deserializeFromAllocateForStorage() unrecognized type '%s'",
                                 getTypeName(type).c_str());
    }
}
*/

/**
 * Serialize this NValue to the provided SerializeOutput
 */

  /*
inline void NValue::serializeTo(SerializeOutput &output) const {
    const ValueType type = getValueType();
    switch (type) {
    case VALUE_TYPE_VARCHAR:
    case VALUE_TYPE_VARBINARY:
    {
        if (isNull()) {
            output.writeInt(OBJECTLENGTH_NULL);
            break;
        }
        const int32_t length = getObjectLength_withoutNull();
        if (length <= OBJECTLENGTH_NULL) {
            throwDynamicSQLException("Attempted to serialize an NValue with a negative length");
        }
        output.writeInt(static_cast<int32_t>(length));

        // Not a null string: write it out
        output.writeBytes(getObjectValue_withoutNull(), length);

        break;
    }
    case VALUE_TYPE_TINYINT: {
        output.writeByte(getTinyInt());
        break;
    }
    case VALUE_TYPE_SMALLINT: {
        output.writeShort(getSmallInt());
        break;
    }
    case VALUE_TYPE_INTEGER: {
        output.writeInt(getInteger());
        break;
    }
    case VALUE_TYPE_TIMESTAMP: {
        output.writeLong(getTimestamp());
        break;
    }
    case VALUE_TYPE_BIGINT: {
        output.writeLong(getBigInt());
        break;
    }
    case VALUE_TYPE_DOUBLE: {
        output.writeDouble(getDouble());
        break;
    }
    case VALUE_TYPE_DECIMAL: {
        output.writeLong(getDecimal().table[1]);
        output.writeLong(getDecimal().table[0]);
        break;
    }
    default:
        throwDynamicSQLException( "NValue::serializeTo() found a column "
                "with ValueType '%s' that is not handled", getValueTypeString().c_str());
    }
}
  */

/** Reformat an object-typed value from its inlined form to its
 *  allocated out-of-line form, for use with a wider/widened tuple
 *  column.  Use the pool specified by the caller, or the temp string
 *  pool if none was supplied. **/
   /*
inline CUDAH void NValue::allocateObjectFromInlinedValue(Pool* pool = NULL)
{
    if (m_valueType == VALUE_TYPE_NULL || m_valueType == VALUE_TYPE_INVALID) {
        return;
    }

    assert(m_valueType == VALUE_TYPE_VARCHAR || m_valueType == VALUE_TYPE_VARBINARY);
    assert(m_sourceInlined);

    if (isNull()) {
        *reinterpret_cast<void**>(m_data) = NULL;
        // serializeToTupleStorage fusses about this inline flag being set, even for NULLs
        setSourceInlined(false);
        return;
    }


    if (pool == NULL) {
      pool = getTempStringPool();//ここの修正も行う
    }

    // When an object is inlined, m_data is a direct pointer into a tuple's inline storage area.
    char* source = *reinterpret_cast<char**>(m_data);

    // When it isn't inlined, m_data must contain a pointer to a StringRef object
    // that contains that same data in that same format.

    int32_t length = getObjectLength_withoutNull();
    // inlined objects always have a minimal (1-byte) length field.
    StringRef* sref = StringRef::create(length + SHORT_OBJECT_LENGTHLENGTH, pool);
    char* storage = sref->get();
    // Copy length and value into the allocated out-of-line storage
    ::memcpy(storage, source, length + SHORT_OBJECT_LENGTHLENGTH);
    setObjectValue(sref);
    setSourceInlined(false);
}
   */

inline CUDAH bool GNValue::isNull() const {
    if (getValueType() == VALUE_TYPE_DECIMAL) {
        TTInt min;
        min.SetMin();
        return getDecimal() == min;
    }

    return m_data[13] == OBJECT_NULL_BIT;
}

/*
inline bool NValue::isNaN() const {
    if (getValueType() == VALUE_TYPE_DOUBLE) {
        return std::isnan(getDouble());
    }
    return false;
}
*/

// general full comparison
/*
inline CUDAH NValue NValue::op_equals(const NValue rhs) const {
    return compare(rhs) == 0 ? getTrue() : getFalse();
}

inline CUDAH NValue NValue::op_notEquals(const NValue rhs) const {
    return compare(rhs) != 0 ? getTrue() : getFalse();
}

inline CUDAH NValue NValue::op_lessThan(const NValue rhs) const {
    return compare(rhs) < 0 ? getTrue() : getFalse();
}

inline CUDAH NValue NValue::op_lessThanOrEqual(const NValue rhs) const {
    return compare(rhs) <= 0 ? getTrue() : getFalse();
}

inline CUDAH NValue NValue::op_greaterThan(const NValue rhs) const {
    return compare(rhs) > 0 ? getTrue() : getFalse();
}

inline CUDAH NValue NValue::op_greaterThanOrEqual(const NValue rhs) const {
    return compare(rhs) >= 0 ? getTrue() : getFalse();
}
*/

// without null comparison
inline CUDAH bool GNValue::op_equals_withoutNull(const GNValue rhs) const {
  int temp = compare_withoutNull(rhs);
  if(temp == -3) return false;
  return temp == 0;
}

inline CUDAH bool GNValue::op_notEquals_withoutNull(const GNValue rhs) const {
  int temp = compare_withoutNull(rhs);
  if(temp == -3) return false;
  return temp != 0;
}

inline CUDAH bool GNValue::op_lessThan_withoutNull(const GNValue rhs) const {
  int temp = compare_withoutNull(rhs);
  if(temp == -3) return false;
  return temp < 0;
}

inline CUDAH bool GNValue::op_lessThanOrEqual_withoutNull(const GNValue rhs) const {
  int temp = compare_withoutNull(rhs);
  if(temp == -3) return false;
  return temp <= 0;

}

inline CUDAH bool GNValue::op_greaterThan_withoutNull(const GNValue rhs) const {
  int temp = compare_withoutNull(rhs);
  if(temp == -3) return false;
  return temp > 0;
}

inline CUDAH bool GNValue::op_greaterThanOrEqual_withoutNull(const GNValue rhs) const {
  int temp = compare_withoutNull(rhs);
  if(temp == -3) return false;
  return temp >= 0;
}


} // namespace voltdb

#endif /* NVALUE_HPP_ */
