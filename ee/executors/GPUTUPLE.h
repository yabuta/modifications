#ifndef GPUTUPLE_H
#define GPUTUPLE_H

namespace voltdb{
//1blockでのスレッド数の定義。
#define BLOCK_SIZE_X 256  //outer ,left
#define BLOCK_SIZE_Y 512  //inner ,right

#define SELECTIVITY 1000000000

    enum CONDITIONFLAG{
        II,
        OI,
    };
    

typedef struct _TUPLE {
    int key;
    int val;
} TUPLE;

typedef struct _RESULT {
    int lkey;
    int rkey;
} RESULT;
}

#endif
