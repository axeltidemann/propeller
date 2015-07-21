'''
Import CSV data to pandas.

After careful consideration, the easiest way to get the data from the postgres database was to make a CSV dump and import the resulting file into pandas. This was done using pgadmin3 (installed on skeletor), which is a GUI tool to view and manipulate postgres databases. Listed below are the SQL queries that were issued. Remember to click on the disk button which will save the query to disk instead of outputting it to the program.

action_log:
SELECT user_id, action, status, card_id, tokens, created_at FROM action_log;

cards:
SELECT id, name, type, available_from, available_to, actions FROM cards;

events:
SELECT user_id, name, card_id, timestamp, properties FROM events;
This database was too big to be exported by pgadmin3, so I used psql directly from the command line:
psql -P format=unaligned -P tuples_only -P fieldsep=\; -c "SELECT user_id, name, card_id, timestamp, properties FROM events" > events.csv -U tidemann -d wowboxprod

Note: use the above will probably yield a much more smooth export from the other two databases as well.

For each of these databases, a subset of the available properties were chosen. The ones that were mainly avoided were json-related configuration strings. Two were included: cards.actions (enumerates which actions were possible - this is again found in action_log), and events.properties, which adds to the detail level in the events log.

Author: Axel.Tidemann@telenor.com
'''


import pandas as pd

with pd.HDFStore('data.h5', 'w', complevel=9, complib='blosc') as store:
    cards = pd.read_csv('cards.csv', delimiter=';', parse_dates=[3,4], index_col=0)
    store.put('cards', cards, data_columns=True)
    print 'cards,',

    action_log = pd.read_csv('action_log.csv', delimiter=';', parse_dates=[5], index_col=5, chunksize=50000,
                             dtype={ 'card_id': pd.np.float64 })
    for chunk in action_log:
        store.append('action_log', chunk, data_columns=True)
    print 'action_log,',

    events = pd.read_csv('events.csv', delimiter=';', parse_dates=[3], index_col=3, chunksize=50000,
                         header=None, names=['user_id', 'name', 'card_id', 'timestamp', 'properties'],
                         dtype={ 'card_id': pd.np.float64 })
    for chunk in events:
        store.append('events', chunk, data_columns=True,
                     min_itemsize={ 'properties': 298 })
    print 'events stored in HDF5.'
