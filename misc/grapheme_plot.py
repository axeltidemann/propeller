# Copyright 2016 Telenor ASA, Author: Axel Tidemann

from __future__ import division
import json
import argparse

import plotly
import plotly.graph_objs as go

parser = argparse.ArgumentParser(description='''
Plots the distribution of graphemes, sorted. Also creates a cumulative plot.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'grapheme_counter')
args = parser.parse_args()

graphemes = json.load(open(args.grapheme_counter))
graphemes = sorted(graphemes.items(), key=lambda x: x[1], reverse=True)

x,y = zip(*graphemes)
text = [ 'x={}, cumulative: {}%'.format(i, 100*sum(y[:i])/sum(y)) for i in range(len(y)) ]
data = [ go.Bar(x=x, y=y, text=text) ]

plotly.offline.plot(data, filename='grapheme_plot.html', auto_open=False)
