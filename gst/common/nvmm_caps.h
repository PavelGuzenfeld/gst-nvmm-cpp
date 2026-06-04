/// Shared NVMM pad-template caps string for the nvmm elements (sink, appsrc,
/// convert). Kept in one place so the supported formats / dimensions / framerate
/// ranges can't drift apart between elements.
#pragma once

#define NVMM_CAPS_STRING \
    "video/x-raw(memory:NVMM), " \
    "format=(string){NV12, RGBA, I420, BGRA}, " \
    "width=(int)[1, 8192], height=(int)[1, 8192], " \
    "framerate=(fraction)[0/1, 240/1]"
