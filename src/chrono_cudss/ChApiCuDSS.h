#ifndef CHAPI_CUDSS_H
#define CHAPI_CUDSS_H

#include "chrono/ChVersion.h"
#include "chrono/core/ChPlatform.h"

#if defined(CH_API_COMPILE_CUDSS)
    #define ChApiCuDSS ChApiEXPORT
#else
    #define ChApiCuDSS ChApiIMPORT
#endif


#endif