# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import subprocess
import os
from os.path import join as pjoin

def call(cmd):
    return subprocess.call(cmd, shell=True)


def download_data(comp_name, target_path="data"):
    cwd = os.getcwd()
    target_path = os.path.join(cwd, target_path)
    mkdir(target_path)
    os.chdir(target_path)
    call("kg download -c {}".format(comp_name))
    os.chdir(cwd)


def unzip(file_path, target_path):
    suffix = file_path.split('.')[-1] 
    if suffix == 'zip':
        cmd = 'unzip -q {} -d {}'.format(file_path, target_path)
    elif suffix == '7z':
        cmd = ' 7z x {} -o{}'.format(file_path, target_path)
    else:
        raise Exception('Unsupport format')
    return call(cmd)

def mkdir(*target_path):
    for p in target_path:
        call('mkdir -p {}'.format(p))


def count_file(target_path):
    return len([f for f in os.listdir(target_path) if os.path.isfile(os.path.join(target_path, f))])


class KaggleCLI(object):
    def __init__(self, competition_name):
        super(KaggleCLI, self).__init__()
        self.competition_name = competition_name

    def download_data(self, data_path='data'):
        cwd = os.getcwd()
        target_path = os.path.join(cwd, data_path)
        mkdir(target_path)
        os.chdir(target_path)
        call("kg download -c {}".format(self.competition_name))
        os.chdir(cwd)

    def submit_result(self, result_path, msg=''):
        call('kg submit -c {} -m {} {}'.format(self.competition_name, msg, result_path))


def execute_in(dir_path, func, *args, **kwargs):
    cwd = os.getcwd()
    os.chdir(dir_path)
    try:
        return func(*args, **kwargs)
    finally:
        os.chdir(cwd)


def unzip_all(dir_path):
    for zfile in os.listdir(dir_path):
        if zfile.endswith('.zip'):
            unzip(pjoin(dir_path, zfile), dir_path)