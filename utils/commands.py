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
    return call('unzip -q {} -d {}'.format(file_path, target_path))

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