import csv

max_size = 3626
max_wires = 32
max_tdcs = 32

# Header 
header = ["Pixel"] * (max_wires * max_tdcs)
feature =  []
label = []
for t in range(max_tdcs):
    for w in range(max_wires):
        if t < 11:
            feature.append(1)
            label.append(0)
        elif t < 22:
            feature.append(10)
            label.append(1)
        else:
            feature.append(10000)
            label.append(2)

with open('feature_w.csv', 'w') as featureFile:
    feature_writer = csv.writer(featureFile)
    feature_writer.writerow(header)
    for _ in range(max_size):
        feature_writer.writerow(feature)
featureFile.close()

with open('label_w.csv', 'w') as labelFile:
    label_writer = csv.writer(labelFile)
    label_writer.writerow(header)
    for _ in range(max_size):
        label_writer.writerow(label)
labelFile.close()
