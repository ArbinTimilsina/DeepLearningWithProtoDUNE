// C++ includes
#include <iostream>
#include <fstream>
#include <string>

// ROOT includes
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>

// DeepLearning includes
#include "dune/Protodune/DeepLearning/Tools/DataStruct.h"
#include "dune/Protodune/DeepLearning/Tools/ImageInformation.h"

// Header file
#include "RootToCSV.h"

using namespace std;

RootToCSV :: RootToCSV(const TString input, const TString output, const int minHits)
    : InputFile(input), OutputDirectory(output), MinHits(minHits)
{
}

void RootToCSV::SetInputFile(const TString input)
{
    InputFile = input;
}

void RootToCSV::SetOutputDirectory(const TString output)
{
    OutputDirectory = output;
}

void RootToCSV::SetMinHits(const int minHits)
{
    MinHits = minHits;
}
///////////////////////////////////////////////////////////////////////////////////////
void RootToCSV::MakeCSV()
{

    // Select the Tree
    TFile *FileInput = new TFile(InputFile);

    std::vector<ProtoDuneDL::HitsStruct> * pHitsStruct = 0;
    TTree *TreeInput = (TTree*)FileInput->Get("DataProcessing/tHitsTree");
    TreeInput->SetBranchAddress("vHitsStruct", &pHitsStruct);

    ///////////////////////////////////////////////////////////////////////////////////////////
    unsigned int nEntries = TreeInput->GetEntries();
    cout << endl;
    if(nEntries <= 0)
        {
            cout << "Error! There are no entries in the Tree." << nEntries << endl << endl;
        }
    else
        {
            cout << "Total Entries in the Tree is: " << nEntries << endl << endl;
        }

    cout << "Min hits for beam and not-beam is: " << MinHits << endl;

    ofstream FileOutput[6]; // 3 planes x (Feature + Label)
    TString FileName[6] = {"feature_u.csv", "feature_v.csv", "feature_w.csv", "label_u.csv", "label_v.csv", "label_w.csv"};

    // Save histograms for hits
    TFile *HistogramFile = new TFile("Histograms.root", "RECREATE");
    TH1D *hHits = new TH1D("hHits", "", 200, 0, 100000);
    TH1D *hHitsBeam = new TH1D("hHitsBeam", "", 100, 0, 7000);
    TH1D *hHitsNotBeam = new TH1D("hHitsNotBeam", "", 300, 0, 30000);

    for (unsigned int iFile = 0; iFile < 6; iFile++)
        {
            // Create and open the .csv file
            FileOutput[iFile].open(OutputDirectory + FileName[iFile]);

            // Write the file headers
            for (unsigned int iPixel = 0; iPixel < ProtoDuneDL::MaxTDCs * ProtoDuneDL::MaxWires; iPixel++)
                {
                    FileOutput[iFile] << "Pixel " << iPixel;
                    if(iPixel != ProtoDuneDL::MaxTDCs * ProtoDuneDL::MaxWires - 1)
                        {
                            FileOutput[iFile] << ",";
                        }
                }
            FileOutput[iFile] << endl;
        }

    unsigned int count = 0;
    for(unsigned int iEntry = 0; iEntry < nEntries; iEntry++)
        {
            TreeInput->GetEntry(iEntry);

            unsigned int nHits = pHitsStruct->size();
            hHits->Fill(nHits);

            unsigned int nHitsBeam = 0;
            unsigned int nHitsNotBeam = 0;
            for(unsigned int iHit = 0; iHit < nHits; iHit++)
                {
                    ProtoDuneDL::HitsStruct hHitsStruct = (*pHitsStruct)[iHit];

		    //Just work with TPC 1
                    if(hHitsStruct.tpc_index != 1)
                        {
                            continue;
                        }
                    if(hHitsStruct.origin == ProtoDuneDL::Labels::Beam)
                        {
                            nHitsBeam++;
                        }
                    if(hHitsStruct.origin == ProtoDuneDL::Labels::NotBeam)
                        {
                            nHitsNotBeam++;
                        }
                }
            hHitsBeam->Fill(nHitsBeam);
            hHitsNotBeam->Fill(nHitsNotBeam);

            // Require minimum no. of hits
            if(nHitsBeam < MinHits || nHitsNotBeam < MinHits)
                {
                    continue;
                }
            count++;

            cout << "Event: " << count << endl;
            //cout << "Total hits: " << nHits << endl;
            //cout << "Beam hits: " << nHitsBeam << endl<<endl;
            //cout << "NotBeam hits: " << nHitsNotBeam << endl<<endl;

            static double FeatureMap[ProtoDuneDL::MaxPlanes][ProtoDuneDL::MaxTDCs][ProtoDuneDL::MaxWires];
            static int LabelMap[ProtoDuneDL::MaxPlanes][ProtoDuneDL::MaxTDCs][ProtoDuneDL::MaxWires];
            for (unsigned int iPlane = 0; iPlane < ProtoDuneDL::MaxPlanes; iPlane++)
                {
                    for (unsigned int iTDC = 0; iTDC < ProtoDuneDL::MaxTDCs; iTDC++)
                        {
                            for (unsigned int iWire = 0; iWire < ProtoDuneDL::MaxWires; iWire++)
                                {

                                    FeatureMap[iPlane][iTDC][iWire] = 0.0;
                                    LabelMap[iPlane][iTDC][iWire] = 0;
                                }
                        }
                }

            for(unsigned int iHit = 0; iHit < nHits; iHit++)
                {
                    ProtoDuneDL::HitsStruct hHitsStruct = (*pHitsStruct)[iHit];

                    //Just work with TPC 1
                    if(hHitsStruct.tpc_index != 1)
                        {
                            continue;
                        }

                    unsigned int plane = hHitsStruct.plane_index;
                    unsigned int tdc = hHitsStruct.tdc;

                    //TO DO: Flip wire for V plane?
                    unsigned int wire = hHitsStruct.wire_index;

                    // TO DO: Better downsampling
                    unsigned int tdcNew = ProtoDuneDL::TransformRange(tdc, 0, ProtoDuneDL::TDCsPerPlane[plane], 0, ProtoDuneDL::MaxTDCs);
                    unsigned int wireNew = ProtoDuneDL::TransformRange(wire, 0, ProtoDuneDL::GlobalWiresPerPlane[plane], 0, ProtoDuneDL::MaxWires);

                    double adc = hHitsStruct.adc;
                    if(adc > FeatureMap[plane][tdcNew][wireNew])
                        {
                            FeatureMap[plane][tdcNew][wireNew] = adc;
                            LabelMap[plane][tdcNew][wireNew] = hHitsStruct.origin;
                        }
                }

            for (unsigned int iPlane = 0; iPlane < ProtoDuneDL::MaxPlanes; iPlane++)
                {
                    int iCount = 0;
                    for (unsigned int iTDC = 0; iTDC < ProtoDuneDL::MaxTDCs; iTDC++)
                        {
                            for (unsigned int iWire = 0; iWire < ProtoDuneDL::MaxWires; iWire++)
                                {
                                    FileOutput[iPlane] << FeatureMap[iPlane][iTDC][iWire];
                                    FileOutput[iPlane + 3] << LabelMap[iPlane][iTDC][iWire];
                                    if(iCount != ProtoDuneDL::MaxTDCs * ProtoDuneDL::MaxWires - 1)
                                        {
                                            FileOutput[iPlane] << ",";
                                            FileOutput[iPlane + 3] << ",";
                                        }
                                    iCount++;
                                }
                        }
                    FileOutput[iPlane] << endl;
                    FileOutput[iPlane + 3] << endl;
                }
        }

    // Close the output file
    for (unsigned int iFile = 0; iFile < 6; iFile++)
        {
            FileOutput[iFile].close();
        }
    HistogramFile->Write();
    HistogramFile->Close();
}

int main(int argc, char* argv[])
{
    if(argc != 4)
        {
            cout << "Error! Please provide 3 arguments: input file, output directory name, and min beam hits." << endl;
            return 1;
        }

    RootToCSV *myRootToCSV = new RootToCSV();
    myRootToCSV->SetInputFile(argv[1]);
    myRootToCSV->SetOutputDirectory(argv[2]);
    myRootToCSV->SetMinHits(atoi(argv[3]));
    myRootToCSV->MakeCSV();

    return 0;
}
