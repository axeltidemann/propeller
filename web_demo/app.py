'''
Code for the Caffe web demo. Original source from http://caffe.berkeleyvision.org. This code requires that 
caffe is installed in $HOME, and that PYTHONPATH points to the same location.

Authors: (listed above) and Axel.Tidemann@telenor.com
'''

import os
import time
import cPickle
from collections import namedtuple
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import random
import time

from functools import wraps
from flask import request, Response

import redis

red = redis.Redis("localhost")
pubsub = red.pubsub(ignore_subscribe_messages=True)
pipe = red.pipeline()

UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# For the position of the word webs
OFFSET = 800

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

@app.route('/images/categories')
@requires_auth
def categories():
    result = red.keys('prediction:web:category:*')
    result = sorted([ cat[cat.rfind(':')+1:] for cat in result ], key=lambda s: s.lower())
    return flask.render_template('categories.html', result=result)

@app.route('/images/categories/<path:category>/')
@requires_auth
def images(category):
    result = red.zrevrangebyscore('prediction:web:category:{}'.format(category),
                                  np.inf, 0, start=0, num=25)
    result = [ unicode(url, 'utf-8') for url in result ]
    return flask.render_template('images.html', category=category, result=result)

def wait_for_prediction(user, path):
    key = 'prediction:{}:{}'.format(user, path)
    result = red.hgetall(key)
    if not result:
        pubsub.psubscribe('__keyspace*__:{}'.format(key))
        for _ in pubsub.listen():
            pubsub.punsubscribe('__keyspace*__:{}'.format(key))
            return red.hgetall(key)
    return result

def generic_result(result):
    if eval(result['OK']):
        return (eval(result['OK']), eval(result['maximally_accurate']), eval(result['maximally_specific']),
                eval(result['computation_time']))
    return (False, 'Something went wrong when classifying the image.')

def specific_result(result):
    if eval(result['OK']):
        return eval(result['maximally_specific'])[0][0]
    return 'Something went wrong when classifying the image.'
    
@app.route('/images/prediction/<path:user>/<path:path>')
@requires_auth
def prediction(user, path):
    return specific_result(wait_for_prediction(user, path))

@app.route('/images/prediction/<path:user>/category/<path:category>')
@requires_auth
def images_in_category(user, category):
    return '\n'.join(red.zrevrangebyscore('prediction:{}:category:{}'.format(user, category), np.inf, 0))

@app.route('/images/classify', methods=['POST'])
@requires_auth
def classify():
    my_file = StringIO.StringIO(request.files['file'].read())

    i = 0
    for line in my_file:
        task = {'user': request.form['user'], 'path': line.strip()}
        if i == 0: 
            pipe.rpush('classify', cPickle.dumps(task))
        else: 
            pipe.lpush('classify', cPickle.dumps(task))

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
    red.rpush('classify', cPickle.dumps({'user': 'web', 'path': imageurl}))

    result = generic_result(wait_for_prediction('web', imageurl))
    return flask.render_template(
        'classify_image.html', has_result=True, result=result, imagesrc=imageurl)

@app.route('/images/classify_upload', methods=['POST'])
@requires_auth
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)
        red.rpush('classify', cPickle.dumps({'user': 'web', 'path': filename}))
        
        result = generic_result(wait_for_prediction('web', filename))

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'classify_image.html', has_result=True,
            result=(False, 'Cannot open uploaded image.'))
        

    #result = app.clf.classify_image(image)
    return flask.render_template(
        'classify_image.html', has_result=True, result=result,
        imagesrc=embed_image_html(image))



def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    opts, args = parser.parse_args()
    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
