'''
Web frontend for the image classifier. Posts images to be classified to the redis server,
which the workers read from.

Author: Axel.Tidemann@telenor.com, Cyril.Banino-Rokkones@telenor.com
'''

import os
import time
import cPickle as pickle
from collections import namedtuple
import datetime
import logging
import argparse
import cStringIO as StringIO
import urllib
import random
from functools import wraps

from flask import request, Response
import flask
import werkzeug
import redis
import tornado.wsgi
import tornado.httpserver


# For the position of the word webs
OFFSET = 800

# Timeout seconds for waiting on the redis key
TIMEOUT = 5

# Obtain the flask app object
app = flask.Flask(__name__)

@app.template_filter('split_path')
def split_path(path):
    full_path = ''
    hrefs = []
    for s in path.split('/'):
        if len(s):
            full_path += '/{}'.format(s)
            hrefs.append([s, full_path])

    hrefs.insert(0, ['~', '/'])
    return hrefs

@app.template_filter('commastrip')
def commastrip(path):
    return path[:path.find(',')] if ',' in path else path

# Remove caching
@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response

def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == 'telenor' and password == 'research'

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
        'Ask for credentials.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

@app.route('/embeddings')
@app.route('/embeddings/wow') # To remove when more categories are included
@requires_auth
def embeddings():
    return flask.render_template('wow.html', has_result=False)

@app.route('/embeddings/wow/<path:bu>')
@requires_auth
def show_word(bu):
    return flask.render_template("network.html", bu=bu)

def get_neighbors(word, nodes, links, pos, level, bu):
    neighbors = red.hgetall(str("wow:"+bu+":") + word)
    neighbors = sorted(neighbors.items(), key=lambda x: x[1])[:7]
    for n in neighbors:
        name = n[0]
        if name not in pos and level > 0:
            nodes.append({"name":name, "x":OFFSET*random.random() , "y":OFFSET*random.random(), "distan\
ce":round(float(n[1]), 2), "to":word})
            pos[name] = len(nodes)-1
        if name in pos:
            links.append({"source":pos[word],"target":pos[name],"value":1.0-float(n[1])})

    if level == 0:
        return

    for n in neighbors:
        get_neighbors(str(n[0]), nodes, links, pos, level-1, bu)


@app.route('/embeddings/wow/json/<path:bu>/<path:word>')
@requires_auth
def get_json(bu, word):
    word = word.strip().lower().encode("utf-8")

    if word == "_":
        word = red.srandmember('wow:'+bu+':vocab')

    tmp = {}
    nodes = []
    nodes.append({"name": word, "x":OFFSET*random.random() , "y":OFFSET*random.random(), "distance":0.0\
, "to":"self"})
    tmp[word] = 0
    links = []
    get_neighbors(word, nodes, links, tmp, 2, bu)
    return flask.jsonify({"nodes":nodes, "links":links})

@app.route('/images')
@requires_auth
def classify_image():
    return flask.render_template('classify_image.html', has_result=False)

@app.route('/images/restful_api')
@requires_auth
def restful():
    return flask.render_template('restful.html')

@app.route('/images/categories')
@requires_auth
def categories():
    result = red.keys('archive:web:category:*')
    result = sorted([ cat[cat.rfind(':')+1:] for cat in result ],
                    key=lambda s: s.lower())
    return flask.render_template('categories.html', result=result)


def get_images_from_category(category, num=-1, group='web'):
    result = red.zrevrangebyscore('archive:{}:category:{}'.format(group, category),
                                  1, 0, start=0, num=num, withscores=True)
    return [ (unicode(url, 'utf-8'), score) for url, score in result ]
    

@app.route('/images/categories/<path:category>/')
@requires_auth
def images(category):
    return flask.render_template('images.html', category=category, result=get_images_from_category(category))

def wait_for_prediction(group, path):
    key = 'archive:{}:{}'.format(group, path)
    t0 = time.time()
    while True:
        if red.exists(key):
            return red.hgetall(key)
        if time.time() - t0 > TIMEOUT:
            return {'OK': 'False'}
        time.sleep(.1)

def parse_result(result):
    if eval(result['OK']):
        return (eval(result['OK']), eval(result['predictions']), eval(result['computation_time']))
    return (False, 'Something went wrong when classifying the image.')
    
@app.route('/images/archive/<path:group>/<path:path>')
@requires_auth
def prediction(group, path):
    return wait_for_prediction(group, path)['predictions']

@app.route('/images/archive/<path:group>/category/<path:category>')
@requires_auth
def images_in_category(group, category):
    return '\n'.join(red.zrevrangebyscore('archive:{}:category:{}'.format(group, category), 1, 0))

@app.route('/images/classify', methods=['POST'])
@requires_auth
def classify():
    my_file = StringIO.StringIO(request.files['file'].read())

    i = 0
    for line in my_file:
        task = {'group': request.form['group'], 'path': line.strip()}
        if i == 0: 
            pipe.rpush(args.queue, pickle.dumps(task))
        else: 
            pipe.lpush(args.queue, pickle.dumps(task))

        i += 1
        if i % 10000 == 0:
            pipe.execute()
            logging.info('Piping 10K items to redis.')

    pipe.execute()
    return '{} images queued for classification.'.format(i)

@app.route('/images/classify_url', methods=['GET'])
@requires_auth
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    red.rpush(args.queue, pickle.dumps({'group': 'web', 'path': imageurl})) # SPECS COMMON!

    prediction = wait_for_prediction('web', imageurl)
    result = parse_result(prediction)
    similar = get_images_from_category(result[1][0][0], 10)
    return flask.render_template(
        'classify_image.html', has_result=True, result=result, imagesrc=imageurl, 
        similar=similar)

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-d', '--debug',
    help="enable debug mode",
    action="store_true", default=False)
parser.add_argument(
    '-p', '--port',
    help="which port to serve content on",
    type=int, default=5000)
parser.add_argument(
    '-rs', '--redis_server',
    help='the redis server address',
    default='localhost')
parser.add_argument(
    '-rp', '--redis_port',
    help='the redis port',
    default='6379')
parser.add_argument(
    '-q', '--queue',
    help='redis queue to post image classification tasks to',
    default='classify')

args = parser.parse_args()

red = redis.StrictRedis(args.redis_server, args.redis_port)
pubsub = red.pubsub(ignore_subscribe_messages=True)
pipe = red.pipeline()

if args.debug:
    app.run(debug=True, host='0.0.0.0', port=args.port)
else:
    start_tornado(app, args.port)
