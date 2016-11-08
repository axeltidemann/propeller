# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import glob
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../training')))

import plotly

from transfer_stats import evaluate

parser = argparse.ArgumentParser(description='''
Creates a report that shows Mudah performance.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'test_folder',
    help='Folder with Inception states for testing')
parser.add_argument(
    'models',
    help='Models to evaluate',
    nargs='+')
parser.add_argument(
    '--top_k',
    help='How many to consider',
    type=int,
    default=3)
parser.add_argument(
    '--categories',
    help='Path to categories JSON file',
    default='categories.json')
parser.add_argument(
    '--gpu',
    help='Which GPU to use for inference. Empty string means no GPU.',
    default='')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
h5_files = sorted(glob.glob('{}/*'.format(args.test_folder)))

categories = json.load(open(args.categories))

html = '''
<html>
<head>
<title>Mudah report on classifier performance</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
'''

for model in args.models:
    figure, accuracy, top_k_accuracy = evaluate(model, h5_files, args.top_k, categories)
    div = plotly.offline.plot_mpl(figure, show_link=False, auto_open=False,
                                  output_type='div', include_plotlyjs=False)

    html += '{}<br>accuracy: {}. top {} accuracy: {}.<br>{}<br>'.format(os.path.basename(model),
                                                                        accuracy, args.top_k, top_k_accuracy, div)
html += '''
</body>
</html>
'''

with open('report.html', 'w') as report:
    report.write(html)
