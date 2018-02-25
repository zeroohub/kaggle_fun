# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import subprocess
import os


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


def make_data_dir(data_path, sample_path):
    mkdir(os.path.join(data_path, 'train'))
    mkdir(os.path.join(data_path, 'test'))
    mkdir(os.path.join(data_path, 'valid'))

    mkdir(os.path.join(sample_path, 'train'))
    mkdir(os.path.join(sample_path, 'test'))
    mkdir(os.path.join(sample_path, 'valid'))

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

    def submit_result(self, result_path):
        call('kg')


def execute_in(dir_path, func, *args, **kwargs):
    cwd = os.getcwd()
    os.chdir(dir_path)
    try:
        return func(*args, **kwargs)
    finally:
        os.chdir(cwd)
