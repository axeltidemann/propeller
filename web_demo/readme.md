
# Telenor Research web demos

## Requirements

The demo web server runs on Python with some dependencies, to install them run `pip install -r requirements.txt`. 

## Run

The web server needs to know the address of the redis server. To get the available options: 

```
python app.py -h 
```

To make it serve on port 80, you must run it in `sudo` mode. An example:

```
sudo python app.py -p 80 -rs redis_server_address
```
