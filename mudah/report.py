# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import glob
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../training')))

import plotly
import plotly.graph_objs as go
from plotly.offline.offline import _plot_html
from collections import Counter, defaultdict

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

<style>
img{
  height: 300px;
}

th{
  text-align: left;
}
</style>


</head>
<body>

<table>
'''

for model in args.models:
    plotly_data, accuracy, top_k_accuracy, stats = evaluate(model, h5_files, args.top_k, categories)

    layout= go.Layout(
        showlegend=False,
        hovermode='closest')

    figure = go.Figure(data=plotly_data, layout=layout)

    plot_html, plotdivid, width, height = _plot_html(figure, False, '', True, '100%', '100%', False)
    
    html += '<tr><td colspan=2><h3>{}</h3>Overall accuracy: {} Overall top-{} accuracy: {}</td></tr><tr><td>{}</td>'.format(os.path.basename(model),
                                                                                                       accuracy, args.top_k,
                                                                                                       top_k_accuracy, plot_html)

    html += '''
    <td>
    <img id="hover-image-'''+str(plotdivid)+'''" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">
    <div id="hoverinfo-'''+str(plotdivid)+'''"></div>
    </td>
    </tr>

    <script>
    var myPlot = document.getElementById("''' + str(plotdivid) + '''"),
    hoverInfo = document.getElementById("hoverinfo-'''+str(plotdivid)+'''"),
    hoverImage = document.getElementById("hover-image-'''+str(plotdivid)+'''");

    myPlot.on('plotly_hover', function(data){
        data.points.map(function(d){
           var info=JSON.parse(d.data.text[d.pointNumber]);
           hoverImage.src="http://52.77.195.194:8080/static"+info.path;
           if( d.data.name == info.prediction ){
               hoverInfo.innerHTML = 'Correctly classified as <b>' + info.prediction +'</b>, score <b>' + d.y.toPrecision(3) + '</b>';
           } else {
               hoverInfo.innerHTML = 'Was in the <b>'+d.data.name+'</b> category, wrongly classified as <b>'+info.prediction+'</b> with score <b>' + d.y.toPrecision(3) + '</b>';
           }
        });
    });

    </script>

    <tr><td colspan=2>'''
        
    for category, accuracy, top_k_accuracy, correct_confidence, wrong_categories, wrong_scores, wrong_paths in sorted(stats, key=lambda x: x[1], reverse=True):
        html += '<b>{}</b> Accuracy: {}% Top-{} accuracy: {}% Correct confidence: {}'.format(category, 100*float(accuracy), args.top_k,
                                                                                             100*float(top_k_accuracy), correct_confidence)

        categories_counter = Counter(wrong_categories)
        labels, values = zip(*categories_counter.items())
        
        figure = {
            'data': [{'labels': labels, 'values': values, 'type': 'pie'}],
            'layout': {'title': 'Wrongly categorized pictures ({}%)'.format(100*(1-float(accuracy))) , 'showlegend': False}
        }

        plot_html, plotdivid, width, height = _plot_html(figure, False, '', True, 400, 400, False)

        html += plot_html
        
        errors = defaultdict(list)
        for c,v,p in zip(wrong_categories, wrong_scores, wrong_paths):
            errors[c].append((v,p))

        plotly_data = []
        for i, (wrong_category, data) in enumerate(errors.iteritems()):
            values, paths = zip(*data)
            
            plotly_data.append(go.Scatter(
                x=[i]*len(values),
                y=values,
                mode='markers',
                hoverinfo='name+y',
                name=wrong_category,
                text=[ json.dumps({ 'path': path, 'prediction': wrong_category }) for path in paths ]))

        labels = errors.keys()
        bandxaxis = go.XAxis(
            ticktext=labels,
            ticks='',
            tickvals=range(len(labels)))
            
        layout= go.Layout(
            showlegend=False,
            hovermode='closest',
            xaxis=bandxaxis)

        figure = go.Figure(data=plotly_data, layout=layout)

        plot_html, plotdivid, width, height = _plot_html(figure, False, '', True, 400, 400, False)
        
        html += '<br>'+plot_html +'''

    <img id="scatimage-'''+str(plotdivid)+'''" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">
    <div id="scatinfo-'''+str(plotdivid)+'''"></div>
        
    <script>

    var scatPlot = document.getElementById("''' + str(plotdivid) + '''"),
    scatInfo = document.getElementById("scatinfo-'''+str(plotdivid)+'''"),
    scatImage = document.getElementById("scatimage-'''+str(plotdivid)+'''");

    scatPlot.on('plotly_hover', function(data){
        data.points.map(function(d){
           var info=JSON.parse(d.data.text[d.pointNumber]);
           scatImage.src="http://52.77.195.194:8080/static"+info.path;
           scatInfo.innerHTML = 'Classified as <b>' + info.prediction +'</b>, score <b>' + d.y.toPrecision(3) + '</b>';
        });
    });

    </script>
        
    </td>
    </tr>
    
    '''
    
html+='''
</table>

</body>
</html>
'''

with open('report.html', 'w') as report:
    report.write(html)
