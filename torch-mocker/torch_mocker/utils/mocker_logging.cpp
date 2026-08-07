// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include "mocker_logging.h"
#include <cstdlib>  // 包含 getenv 函数的头文件

bool Vlog() {
    const char* envVarValue = std::getenv("VLOG_MOCKER_LEVEL");
    return envVarValue != nullptr;
}
