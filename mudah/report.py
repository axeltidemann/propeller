# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import glob
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../training')))

import plotly
import plotly.graph_objs as go

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

    from plotly.offline.offline import _plot_html

    plot_html, plotdivid, width, height = _plot_html(figure, False, '', True, '100%', '100%', False)
    
    html += '<tr><td colspan=2><h3>{}</h3>accuracy: {} top-{} accuracy: {}</td></tr><tr><td>{}</td>'.format(os.path.basename(model),
                                                                                                       accuracy, args.top_k,
                                                                                                       top_k_accuracy, plot_html)

    html += '''
    <td>
    <img id="hover-image-'''+str(plotdivid)+'''" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">
    <div id="hoverinfo-'''+str(plotdivid)+'''"></div>
    </td>
    </tr>

    <tr><td colspan=2>

    <table width=800>
    <tr><th>Category</th><th>Accuracy</th><th>Top-'''+str(args.top_k)+''' accuracy</th><th>Correct confidence</th></tr>'''
    
    for row in sorted(stats, key=lambda x: x[1], reverse=True):
        html += '<tr>'
        for element in row:
            html += '<td>{}</td>'.format(element)
        html += '</tr>'
    html += '''
    </table>
    
    </td>
    </tr>
    
    <script>
    var myPlot = document.getElementById("''' + str(plotdivid) + '''"),
    hoverInfo = document.getElementById("hoverinfo"'''+str(plotdivid)+'''"),
    hoverImage = document.getElementById("hover-image"'''+str(plotdivid)'''+");

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
    
    '''
    
html+='''
</table>

</body>
</html>
'''

with open('report.html', 'w') as report:
    report.write(html)
