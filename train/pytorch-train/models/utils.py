#!/usr/bin/python
# encoding: utf-8


def data_parallel(model, input, ngpu):
    return model(input)
