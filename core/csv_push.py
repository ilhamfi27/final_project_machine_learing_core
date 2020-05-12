import csv
import os.path

class CSVPush():

    def __init__(self, file_name, row_names):
        self.file_name = file_name
        self.row_names = row_names

    def push(self, data):
        with open(self.file_name, 'a', newline='') as file:
            file_empty = os.stat(self.file_name).st_size == 0
            writer = csv.DictWriter(file, fieldnames = self.row_names)
            if file_empty:
                writer.writeheader()
            writer.writerow(dict(zip(self.row_names, data)))
