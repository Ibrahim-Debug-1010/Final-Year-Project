import qwiic_as726x
import time
import csv

sensor = qwiic_as726x.QwiicAS726x()

if not sensor.connected:
    print("AS726x not detected. Check connections.")
    exit()

sensor.begin()
print("AS726x Initialized. Starting data collection...")

filename = "datasets/glucose_data.csv"
header = ["610", "680", "730", "760", "810", "860", "glucose"]

with open(filename, mode="a", newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(header)

    while True:
        sensor.take_measurements()
        vals = [
            sensor.get_calibrated_value(0),
            sensor.get_calibrated_value(1),
            sensor.get_calibrated_value(2),
            sensor.get_calibrated_value(3),
            sensor.get_calibrated_value(4),
            sensor.get_calibrated_value(5),
        ]
        print("Spectral values:", vals)
        glucose = input("Enter corresponding glucose reading (mg/dL): ")

        if glucose.lower() == "q":
            print("Data collection ended.")
            break

        writer.writerow(vals + [glucose])
        print("Saved entry.\n")
