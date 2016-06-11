#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import random
import csv

ampls = np.linspace(25.0, 100.0, 4)
freqs = np.logspace(2, 5, 20, base=10)
mt_list = []
for item in itertools.product(ampls, freqs):
    mt_list.append(item)
random.shuffle(mt_list)

save_path = 'mt_list_postmortem_2016-06-10.csv'
labels = '!Intensity', 'Frequency'
units = '!rad/s', 'Hz'
no_mt = '0.1', '100000000.0'
with open(save_path, 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=str('\t'))
    csv_writer.writerow(labels)
    csv_writer.writerow(units)
    csv_writer.writerow(no_mt)
    for item in mt_list:
        csv_writer.writerow(item)
