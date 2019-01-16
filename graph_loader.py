import csv
import datetime
import progressbar

import graph_model


TOTAL_MAX = 173179759
MAX = 1000000
i = 0


def get_widgets():
    return [
        'Loading: ', progressbar.Percentage(),
        ' | ', progressbar.AdaptiveETA()
    ]


def load_graph_from_csv(path, predicate):
    global i
    graph = graph_model.Graph()
    with open(path) as file:
        reader = csv.DictReader(file)

        bar = progressbar.ProgressBar(widgets=get_widgets(), max_value=TOTAL_MAX).start()
        i = 0
        for row in reader:
            if predicate(row):
                from_coordinates = (float(row['pickup_latitude']), float(row['pickup_longitude']))
                to_coordinates = (float(row['dropoff_latitude']), float(row['dropoff_longitude']))
                passangers = int(row['passenger_count'])
                timestamp = datetime.datetime.strptime(row['pickup_datetime'], '%Y-%m-%d %H:%M:%S')
                graph.add_coordinates(from_coordinates, to_coordinates, passangers, timestamp)

            i += 1
            bar.update(i)
    bar.finish()
    return graph


if __name__ == '__main__':
    path = 'trip_data_filtered.csv'
    path_to_save = ''

    name_predicate = {
        'graph_jan.gpickle': lambda row: datetime.datetime.strptime(row['pickup_datetime'],
                                                                    '%Y-%m-%d %H:%M:%S').month == 1,
        'graph_sep.gpickle': lambda row: datetime.datetime.strptime(row['pickup_datetime'],
                                                                    '%Y-%m-%d %H:%M:%S').month == 9,
        'graph.gpickle': lambda row: i % 12 == 0
    }

    for name, predicate in name_predicate.items():
        print 'Loading ', name
        graph_model = load_graph_from_csv(path, predicate)
        print 'Saving ', name
        graph_model.save_graph(path_to_save + name)

