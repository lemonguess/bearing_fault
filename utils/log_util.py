#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024-03-17 1:22
@Author  : lxc
@File    : og_util.py
@Desc    :

"""
import os, sys, time
log_path = './logs'
if not os.path.exists(log_path):
    os.mkdir(log_path)
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": '[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] '
                      '[%(levelname)s]- %(message)s'
        }
    },
    "handlers": {
        "all": {
            "level": "DEBUG",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "default",
            "filename": os.path.join(log_path, 'all-{}.log'.format(time.strftime('%Y-%m-%d'))),
            "when": "midnight",
            "backupCount": 1,
            'encoding': 'utf-8',  # 设置默认编码
        },
        'error': {
            'level': 'ERROR',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': os.path.join(log_path, 'error-{}.log'.format(time.strftime('%Y-%m-%d'))),
            'backupCount': 1,
            'formatter': 'default',  # 输出格式
            'encoding': 'utf-8',  # 设置默认编码
            "when": "midnight",
        },
        'info': {
            'level': 'INFO',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': os.path.join(log_path, 'info-{}.log'.format(time.strftime('%Y-%m-%d'))),
            'backupCount': 1,
            'formatter': 'default',
            'encoding': 'utf-8',  # 设置默认编码
            "when": "midnight",
        },
        "console_handler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["all", "error", "info", "console_handler"],
        'propagate': True
    }
}