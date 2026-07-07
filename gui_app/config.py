# -*- coding: utf-8 -*-
import os
from PyQt5.QtCore import QSettings

def _settings():
    return QSettings("NhepMieng", "MuseTalk_App")

def set(key: str, value):
    s = _settings()
    s.setValue(key, value)

def get_str(key: str, default="") -> str:
    s = _settings()
    return str(s.value(key, default))

def get_int(key: str, default=0) -> int:
    s = _settings()
    v = s.value(key, default)
    try:
        return int(v)
    except:
        return default

def get_float(key: str, default=0.0) -> float:
    s = _settings()
    v = s.value(key, default)
    try:
        return float(v)
    except:
        return default

def get_bool(key: str, default=False) -> bool:
    s = _settings()
    v = s.value(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == 'true'
    try:
        return bool(int(v))
    except:
        return default

DEFAULTS = {
    'sadtalker_root'   : 'd:/nhep',
    'checkpoint_path'  : 'd:/nhep/models',
    'ffmpeg_path'      : 'd:/nhep/ffmpeg/ffmpeg.exe',
    'output_dir'       : 'd:/nhep/results',
    'device'           : 'cuda',
    'default_voice'    : 'vi-VN-HoaiMyNeural',
    'bbox_shift'       : 0,
    'extra_margin'     : 10,
    'parsing_mode'     : 'jaw',
    'left_cheek_width' : 90,
    'right_cheek_width': 90,
    'use_enhancer'     : True,
    'batch_size'       : 8,
    'tts_rate'         : 0,
}

def sadtalker_root()  -> str:  return get_str('sadtalker_root', DEFAULTS['sadtalker_root'])
def checkpoint_path() -> str:  return get_str('checkpoint_path', DEFAULTS['checkpoint_path'])
def ffmpeg_path()     -> str:  return get_str('ffmpeg_path', DEFAULTS['ffmpeg_path'])
def output_dir()      -> str:  return get_str('output_dir', DEFAULTS['output_dir'])
def device()          -> str:  return get_str('device', DEFAULTS['device'])
def default_voice()   -> str:  return get_str('default_voice', DEFAULTS['default_voice'])
def bbox_shift()      -> int:  return get_int('bbox_shift', DEFAULTS['bbox_shift'])
def extra_margin()    -> int:  return get_int('extra_margin', DEFAULTS['extra_margin'])
def parsing_mode()    -> str:  return get_str('parsing_mode', DEFAULTS['parsing_mode'])
def left_cheek_width() -> int: return get_int('left_cheek_width', DEFAULTS['left_cheek_width'])
def right_cheek_width()-> int: return get_int('right_cheek_width', DEFAULTS['right_cheek_width'])
def use_enhancer()    -> bool: return get_bool('use_enhancer', DEFAULTS['use_enhancer'])
def batch_size()      -> int:  return get_int('batch_size', DEFAULTS['batch_size'])
def tts_rate()        -> int:  return get_int('tts_rate', DEFAULTS['tts_rate'])
