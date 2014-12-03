/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_voltdb_utils_PosixAdvise */

#ifndef _Included_org_voltdb_utils_PosixAdvise
#define _Included_org_voltdb_utils_PosixAdvise
#ifdef __cplusplus
extern "C" {
#endif
#undef org_voltdb_utils_PosixAdvise_POSIX_MADV_NORMAL
#define org_voltdb_utils_PosixAdvise_POSIX_MADV_NORMAL 0L
#undef org_voltdb_utils_PosixAdvise_POSIX_MADV_RANDOM
#define org_voltdb_utils_PosixAdvise_POSIX_MADV_RANDOM 1L
#undef org_voltdb_utils_PosixAdvise_POSIX_MADV_SEQUENTIAL
#define org_voltdb_utils_PosixAdvise_POSIX_MADV_SEQUENTIAL 2L
#undef org_voltdb_utils_PosixAdvise_POSIX_MADV_WILLNEED
#define org_voltdb_utils_PosixAdvise_POSIX_MADV_WILLNEED 3L
#undef org_voltdb_utils_PosixAdvise_POSIX_MADV_DONTNEED
#define org_voltdb_utils_PosixAdvise_POSIX_MADV_DONTNEED 4L
#undef org_voltdb_utils_PosixAdvise_POSIX_FADV_NORMAL
#define org_voltdb_utils_PosixAdvise_POSIX_FADV_NORMAL 0L
#undef org_voltdb_utils_PosixAdvise_POSIX_FADV_RANDOM
#define org_voltdb_utils_PosixAdvise_POSIX_FADV_RANDOM 1L
#undef org_voltdb_utils_PosixAdvise_POSIX_FADV_SEQUENTIAL
#define org_voltdb_utils_PosixAdvise_POSIX_FADV_SEQUENTIAL 2L
#undef org_voltdb_utils_PosixAdvise_POSIX_FADV_WILLNEED
#define org_voltdb_utils_PosixAdvise_POSIX_FADV_WILLNEED 3L
#undef org_voltdb_utils_PosixAdvise_POSIX_FADV_DONTNEED
#define org_voltdb_utils_PosixAdvise_POSIX_FADV_DONTNEED 4L
#undef org_voltdb_utils_PosixAdvise_SYNC_FILE_RANGE_WAIT_BEFORE
#define org_voltdb_utils_PosixAdvise_SYNC_FILE_RANGE_WAIT_BEFORE 1L
#undef org_voltdb_utils_PosixAdvise_SYNC_FILE_RANGE_WRITE
#define org_voltdb_utils_PosixAdvise_SYNC_FILE_RANGE_WRITE 2L
#undef org_voltdb_utils_PosixAdvise_SYNC_FILE_RANGE_WAIT_AFTER
#define org_voltdb_utils_PosixAdvise_SYNC_FILE_RANGE_WAIT_AFTER 4L
#undef org_voltdb_utils_PosixAdvise_SYNC_FILE_RANGE_SYNC
#define org_voltdb_utils_PosixAdvise_SYNC_FILE_RANGE_SYNC 7L
/*
 * Class:     org_voltdb_utils_PosixAdvise
 * Method:    madvise
 * Signature: (JJI)J
 */
JNIEXPORT jlong JNICALL Java_org_voltdb_utils_PosixAdvise_madvise
  (JNIEnv *, jclass, jlong, jlong, jint);

/*
 * Class:     org_voltdb_utils_PosixAdvise
 * Method:    fadvise
 * Signature: (JJJI)J
 */
JNIEXPORT jlong JNICALL Java_org_voltdb_utils_PosixAdvise_fadvise
  (JNIEnv *, jclass, jlong, jlong, jlong, jint);

/*
 * Class:     org_voltdb_utils_PosixAdvise
 * Method:    fallocate
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_org_voltdb_utils_PosixAdvise_fallocate
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     org_voltdb_utils_PosixAdvise
 * Method:    sync_file_range
 * Signature: (JJJI)J
 */
JNIEXPORT jlong JNICALL Java_org_voltdb_utils_PosixAdvise_sync_1file_1range
  (JNIEnv *, jclass, jlong, jlong, jlong, jint);

#ifdef __cplusplus
}
#endif
#endif
