/*
 TIMER
*/
#include <sys/time.h>

#define TIMER_INIT() \
{\
struct timeval tv1,tv2;\
struct timezone tz;\
}\

#define TIMER_BEGIN() \
{\
    gettimeofday(&tv1, &tz);\
}\

#define TIMER_END() \
{\
    gettimeofday(&tv2, &tz);\
}\

#define TIMER_INTERVAL \
(\
    (long long)(tv2.tv_sec  - tv1.tv_sec )*1000000 \
             + (tv2.tv_usec - tv1.tv_usec) \
)\

/*
 MIN_MED_MAX
 */
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

/*
 ALLOCATE
 */
#define allocate(p, n) \
{\
    if((p) != NULL) free(p);\
    (p) = (__typeof__(p)) malloc (sizeof(*(p)) * (n));\
    memset(p, 0, sizeof(*(p)) * (n));\
    if((p) == NULL) fprintf(stderr, "ALLOCATE\n"), exit(1);\
}\

#define deallocate(p) \
{\
    if((p) != NULL) free(p);\
    (p) = NULL;\
}\

/*
 COMPLEX
 */
//typedef double cmpx[2];
//typedef float cmpxf[2];
