#pragma once

#define NVMM_CAPS_STRING \
    "video/x-raw(memory:NVMM), " \
    "format=(string){NV12, RGBA, I420, BGRA}, " \
    "width=(int)[1, 8192], height=(int)[1, 8192], " \
    "framerate=(fraction)[0/1, 240/1]"
