cd ../input_files

rm -rf training/*.csv
scp arbint@dunegpvm03.fnal.gov:/nashome/a/arbint/DuneData/RootFiles/DL-Work/RootToCSV/Training/feature_w.csv training/feature.csv
scp arbint@dunegpvm03.fnal.gov:/nashome/a/arbint/DuneData/RootFiles/DL-Work/RootToCSV/Training/label_w.csv training/label.csv

rm -rf testing/*.csv
scp arbint@dunegpvm03.fnal.gov:/nashome/a/arbint/DuneData/RootFiles/DL-Work/RootToCSV/Testing/feature_w.csv testing/feature.csv
scp arbint@dunegpvm03.fnal.gov:/nashome/a/arbint/DuneData/RootFiles/DL-Work/RootToCSV/Testing/label_w.csv testing/label.csv

rm -rf validation/*.csv
scp arbint@dunegpvm03.fnal.gov:/nashome/a/arbint/DuneData/RootFiles/DL-Work/RootToCSV/Validation/feature_w.csv validation/feature.csv
scp arbint@dunegpvm03.fnal.gov:/nashome/a/arbint/DuneData/RootFiles/DL-Work/RootToCSV/Validation/label_w.csv validation/label.csv
