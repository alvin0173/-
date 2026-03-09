# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 09:16:29 2025

@author: ChenYijian
"""
import os
import logging

level_dict = {'DEBUG': logging.DEBUG,
              'INFO': logging.INFO,
              'WARNING': logging.WARNING
    }

    
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

def init_logger():
    
    # 创建日志器
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    
    # 先移除已有的日志器
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 监控日志
    info_format = logging.Formatter('[ %(asctime)s - %(filename)-8s - %(levelname)s ]: %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
    info_path = os.path.join('method_result.log')
    info_handler = logging.FileHandler(info_path)
    info_handler.setFormatter(info_format)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(InfoFilter())
    logger.addHandler(info_handler)
    
    
    return logger
    
         
    