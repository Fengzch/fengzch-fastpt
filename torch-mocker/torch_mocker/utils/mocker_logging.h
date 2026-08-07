// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MOCKER_LOGGING_H
#define MOCKER_LOGGING_H
#include <iostream>

#define LOG_INFO() printf("LOG(INFO) %s: %d %s\n", __FILE__, __LINE__, __func__)
// 函数声明
bool Vlog();
#endif // MOCKER_LOGGING_H

