#ifndef ROOT_TO_CSV_H
#define ROOT_TO_CSV_H

using namespace std;

class RootToCSV
{
 private :

    TString InputFile;
    TString OutputDirectory;
    unsigned int MinHits;

 public :

    // With default values
    RootToCSV(const TString input = "NONE", const TString output = "NONE", const int minHits = 500);

    // To set values of choice
    void SetInputFile(const TString input);
    void SetOutputDirectory(const TString output);
    void SetMinHits(const int minHits);

    // To make CSV file
    void MakeCSV();
};

#endif
