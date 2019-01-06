////////////////////////////////////////////////////////////////////////
// Class:        DataProcessing
// Module Type:  analyzer
// File:         DataProcessing_module.cc
// Author:       Arbin Timilsina, arbint@bnl.gov
////////////////////////////////////////////////////////////////////////

// Framework includes
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art/Framework/Services/Optional/TFileService.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "messagefacility/MessageLogger/MessageLogger.h"
#include "fhiclcpp/ParameterSet.h"

// LArSoft includes
#include "lardataobj/RecoBase/Hit.h"
#include "larsim/MCCheater/BackTrackerService.h"
#include "larsim/MCCheater/ParticleInventoryService.h"
#include "nusimdata/SimulationBase/MCParticle.h"
#include "larcore/Geometry/Geometry.h"
#include "larcorealg/Geometry/GeometryCore.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesService.h"

// DeepLearning includes
#include "dune/Protodune/DeepLearning/Tools/ImageInformation.h"

// Header file
#include "DataProcessing_module.h"

using namespace std;

namespace ProtoDuneDL
{
    /////////////////////////////////////////////////////////
    //Constructor
    /////////////////////////////////////////////////////////
    DataProcessing::DataProcessing(fhicl::ParameterSet const& parameterSet)
	: EDAnalyzer(parameterSet)
    {
	// Read in the parameters from the .fcl file
	this->reconfigure(parameterSet);
    }


    /////////////////////////////////////////////////////////
    //Destructor
    /////////////////////////////////////////////////////////
    DataProcessing::~DataProcessing() {}

    /////////////////////////////////////////////////////////
    //Reads parameters form the .fcl file
    /////////////////////////////////////////////////////////
    void DataProcessing::reconfigure(fhicl::ParameterSet const& p)
    {
	fHitsModuleLabel = p.get< std::string >("HitsModuleLabel");
	fDebug = p.get< bool >("Debug");
	return;
    }

    /////////////////////////////////////////////////////////
    //Executes once at the beginning of the job
    /////////////////////////////////////////////////////////
    void DataProcessing::beginJob()
    {
	nEvents = 0;

	// Open a basic log file, will overwrite a pre-existing one
	logFile.open("DataProcessing.log");

	// Get local time
	time_t rawtime;
	struct tm * timeinfo;
	time (&rawtime);
	timeinfo = localtime (&rawtime);
	logFile << "DataProcessing_module log file, " << asctime(timeinfo) << endl << endl;

	// Get geometry information
	unsigned int nCryo = geom->Ncryostats(); // 1
	unsigned int nTPC = geom->NTPC(); // 12
	unsigned int nAPA = nCryo * nTPC / 2; // 6
	unsigned int nPlanes = geom->Nplanes(0);
	unsigned int nChannels = geom->Nchannels();

	logFile << "Geometry information for ProtoDUNE:" << endl;
	logFile << "No. of cryostats: " << nCryo << endl;
	logFile << "No. of TPCs: " << nTPC << endl;
	logFile << "No. of APAs: " << nAPA << endl;
	logFile << "No. of Planes: " << nPlanes << endl;
	logFile << "No. of channels in the detector: " << nChannels << endl;
	logFile << "No. of wires per TPC (U, V, W):" << endl;
	for(unsigned int  t = 0; t < nTPC; t++)
	    {
		unsigned int nWiresU = geom->Nwires(0, t); //(plane, tpc)
		unsigned int nWiresV = geom->Nwires(1, t);
		unsigned int nWiresW = geom->Nwires(2, t);

		if(fDebug || t == 0)
		    {
			logFile << "TPC no. " << t << ": (" << nWiresU << ", " << nWiresV << "; " << nWiresW << ")" << endl;
		    }
	    }
	logFile << endl;

	// Get timing information
	auto const* detectorPropertiesService = lar::providerFrom<detinfo::DetectorPropertiesService>();
	double driftVelocity = detectorPropertiesService->DriftVelocity(); //cm/us
	// Sampling rate in us
	double samplingRate = detectorPropertiesService->SamplingRate() * 1E-3;
	// Number of clock ticks per event
	unsigned int clockTicksPerEvent = detectorPropertiesService->NumberTimeSamples();
	// Number of clock ticks per readout window
	unsigned int nTicks = detectorPropertiesService->ReadOutWindowSize();
	// Time per readout window in us
	double readOutWindowSizeTime = nTicks * samplingRate;
	// X-coordinate per tick
	double tickToX = driftVelocity * samplingRate;

	logFile << "Detector properties for ProtoDUNE:" << endl;
	logFile << "Drift velocity (cm/us): " << driftVelocity << endl;
	logFile << "Sampling rate (us): " << samplingRate << endl;
	logFile << "Number of clock ticks per event: " << clockTicksPerEvent << endl;
	logFile << "Number of clock ticks per readout window: " << nTicks << endl;
	logFile << "Time per readout window (us): " << readOutWindowSizeTime << endl;
	logFile << "Tick to x (cm): " << tickToX << endl << endl;

	if(fDebug)
	    {
		logFile << "Global wire information:" << endl;
		for (unsigned int c = 0; c < nCryo; c++)
		    {
			for (unsigned int p = 0; p < nPlanes; p++)
			    {
				for (unsigned int t = 0; t < nTPC; t++)
				    {
					for (unsigned int w = 0; w < geom->Nwires(p, t, c); w++)
					    {
						geo::WireID wireID = {c, t, p, w};
						unsigned int globalWire;
						unsigned int globalPlane;
						ProtoDuneDL::GetGlobalWire(wireID, globalWire, globalPlane);
						logFile << "Cryo: " << c << "; plane: " << p << "; tpc: " << t << "; wire: " << w <<
                                                    "; global plane: " << globalPlane << "; global wire: " << globalWire << endl;
					    }
				    }
			    }
		    }
	    }
	logFile << endl;

	// Access ART's TFileService, which will handle creating and writing histograms and n-tuples
	art::ServiceHandle<art::TFileService> fileServiceHandle;
	tHitsTree = fileServiceHandle->make<TTree>("tHitsTree", "Hits tree");
	tHitsTree->Branch("vHitsStruct", "std::vector<ProtoDuneDL::HitsStruct>", &pHitsStruct);

	if(fDebug)
	    {
		tAllHitsTree = fileServiceHandle->make<TTree>("tAllHitsTree", "All hits tree");
		tAllHitsTree->Branch("vAllHitsStruct", "std::vector<ProtoDuneDL::AllHitsStruct>", &pAllHitsStruct);

		tPrimaryParticleTree = fileServiceHandle->make<TTree>("tPrimaryParticleTree", "Primary particle tree");
		tPrimaryParticleTree->Branch("vPrimaryParticleStruct", "std::vector<ProtoDuneDL::PrimaryParticleStruct>", &pPrimaryParticleStruct);
	    }
    }

    /////////////////////////////////////////////////////////
    //Executes once at the end of the job
    /////////////////////////////////////////////////////////
    void DataProcessing::endJob()
    {
	logFile << "EndJob called, all done!!!" << endl;
    }

    /////////////////////////////////////////////////////////
    //Executes once per event
    /////////////////////////////////////////////////////////
    void DataProcessing::analyze( const art::Event & event )
    {
	// Real data will not have primary particle (label) information
	bool isMC = true;
	if(event.isRealData())
	    {
		logFile << "Running over real data!!" << endl << endl;
		isMC = false;
	    }
	else
	    {
		logFile << "Running over simulation!!" << endl << endl;
	    }

	// Clear the vectors for each event
	pHitsStruct->clear();
	if(fDebug)
	    {
		pAllHitsStruct->clear();
		pPrimaryParticleStruct->clear();
	    }

	// Keep track of event
	nEvents++;
	unsigned int eventId  = event.id().event();
	logFile << endl;
	logFile << "Event no: " << nEvents << "; event id: " << eventId << endl << endl;

	// Get event MC truth information
	// Map of trackID and energy of primary to match to hits
	map<int, unsigned int> mapPrimaryToHits; // trackID and origin
	if(isMC)
	    {
		std::vector< art::Handle< std::vector<simb::MCTruth> > > allMCTruthList;
		event.getManyByType(allMCTruthList);
		for(unsigned int mcl = 0; mcl < allMCTruthList.size(); ++mcl)
		    {
			art::Handle< std::vector<simb::MCTruth> > allMCTruthListHandle = allMCTruthList[mcl];
			for(unsigned int mch = 0; mch < allMCTruthListHandle->size(); ++mch)
			    {
				art::Ptr<simb::MCTruth> mcTruth(allMCTruthListHandle, mch);
				auto truthOrigin = mcTruth->Origin();

				unsigned int nTruthParticles = mcTruth->NParticles();
				logFile << "No. of truth particles: " << nTruthParticles << endl << endl;

				for(unsigned int iParticle = 0; iParticle < nTruthParticles; iParticle++)
				    {
					const simb::MCParticle& particle(mcTruth->GetParticle(iParticle));
					int pdg = particle.PdgCode();
					double energy = particle.E();
					int genTrackId = particle.TrackId();

					// Match this primary muon to a GEANT track and assign GEANT track id
					int geantTrackId = -9999;
					const sim::ParticleList& geantList = particleInventory->ParticleList();
					for (const auto& PartPair : geantList)
					    {
						const simb::MCParticle& geantParticle = *(PartPair.second);
						if((pdg == geantParticle.PdgCode()) &&
						   (fabs(particle.Px() - geantParticle.Px()) < 0.0001) &&
						   (fabs(particle.Py() - geantParticle.Py()) < 0.0001) &&
						   (fabs(particle.Pz() - geantParticle.Pz()) < 0.0001))
						    {
							geantTrackId = geantParticle.TrackId();
							break;
						    }
					    }

					unsigned int primaryOrigin = ProtoDuneDL::Labels::NotBeam;
					if ((truthOrigin == simb::kSingleParticle) && (particle.Process() == "primary"))
					    {
						primaryOrigin = ProtoDuneDL::Labels::Beam;
					    }

					if(fDebug || (iParticle <= 5 && nEvents == 1))
					    {
						logFile << "Particle: " << iParticle << endl;
						logFile << "Origin: " << truthOrigin << endl;
						logFile << "PDG: " << pdg << endl;
						logFile << "Energy: " << energy << endl << endl;
						logFile << "Generator TrackId: " << genTrackId << endl << endl;
						logFile << "GEANT TrackId: " << geantTrackId << endl << endl;
					    }

					if(fDebug)
					    {
						ProtoDuneDL::PrimaryParticleStruct tempParticle;
						tempParticle.event_no = nEvents;
						tempParticle.event_id = eventId;
						tempParticle.origin = primaryOrigin;
						tempParticle.n_particles = nTruthParticles;
						tempParticle.pdg = pdg;
						tempParticle.energy = energy;
						tempParticle.gen_track_id = genTrackId;
						tempParticle.geant_track_id = geantTrackId;
						pPrimaryParticleStruct->push_back(tempParticle);
					    }

					//Skip the particle if it can't be assigned to a GEANT track
					if(geantTrackId == -9999)
					    {
						continue;
					    }
					mapPrimaryToHits.insert(make_pair(geantTrackId, primaryOrigin));
				    }
			    }
		    }
		if(fDebug)
		    {
			tPrimaryParticleTree->Fill();
		    }
	    }

	// Get hit information
	art::Handle< std::vector< recob::Hit > > hitListHandle;
	std::vector< art::Ptr< recob::Hit > > hitList;
	if (event.getByLabel(fHitsModuleLabel, hitListHandle))
	    {
		art::fill_ptr_vector(hitList, hitListHandle);
	    }

	unsigned int nHits = hitList.size();
	logFile << "No. of hits: " << nHits << endl << endl;
	for(unsigned int iHit = 0; iHit < nHits; iHit++)
	    {
		// Integral under the calibrated signal waveform of the hit, in tick x ADC units
		double adc  = hitList[iHit]->Integral();

		// Time of the signal peak, in tick units
		double tdc = hitList[iHit]->PeakTime();

		// Get WireID for the hit (wire with index Cryostat, TPC, Plane, Wire)
		geo::WireID wireID = hitList[iHit]->WireID();
		unsigned int cryostatIndex = wireID.Cryostat;
		unsigned int tpcIndex = wireID.TPC;
		unsigned int planeIndex = wireID.Plane;
		unsigned int wireIndex = wireID.Wire;

		unsigned int globalWireIndex;
		unsigned int globalPlaneIndex;
		ProtoDuneDL::GetGlobalWire(wireID, globalWireIndex, globalPlaneIndex);

		// Get channel no.
		// There are multiple wires per DAQ channel for induction-plane channels in ProtoDUNE
		unsigned int channelNo = geom->PlaneWireToChannel(planeIndex, wireIndex, tpcIndex, cryostatIndex);

		// Get truth information about the hit
		int bestTrackId = -9999;
		double totalHitEnergy = 0.0, bestTrackIdEnergy = 0.0;
		unsigned int pdg = 9999;
		double truthEnergy = 0.0;

		unsigned int hitOrigin = ProtoDuneDL::Labels::Background;
		if(isMC)
		    {
			// Multiple truth can contribute to a hit; we assign hit to trackID with max energy
			unordered_map<int, double> mapIdEnergy;
			for (auto const & ide : backTracker->HitToTrackIDEs(hitList[iHit]))
			    {
				mapIdEnergy[ide.trackID] += ide.energy; // Sum energy from the hit with Geant4 supplied trackID [MeV]
			    }

			for (auto const & contrib : mapIdEnergy)
			    {
				totalHitEnergy += contrib.second; // Sum total energy in these hits
				if (contrib.second > bestTrackIdEnergy) // Find track ID corresponding to max energy
				    {
					bestTrackIdEnergy = contrib.second;
					bestTrackId = contrib.first;
				    }
			    }

			if(totalHitEnergy <= 0.0 || bestTrackId == -9999)
			    {
				continue;
			    }

			// Negative ID means this is EM activity caused by track with the same but positive ID
			if(bestTrackId < 0)
			    {
				bestTrackId = -bestTrackId;
			    }

			auto origin = particleInventory->TrackIdToMCTruth_P(bestTrackId)->Origin();
                    if((origin == simb::kSingleParticle) && (particleInventory->TrackIdToParticle_P(bestTrackId)->Process() == "primary"))
                        {
                            hitOrigin = ProtoDuneDL::Labels::Beam;
                        }
                    else
                        {
                            hitOrigin = ProtoDuneDL::Labels::NotBeam;
                        }

                    pdg = particleInventory->TrackIdToParticle_P(bestTrackId)->PdgCode();
                    truthEnergy = particleInventory->TrackIdToParticle_P(bestTrackId)->E() * 1E-3;
                }

            if(fDebug || (iHit <= 5 && nEvents == 1))
                {
                    logFile << "Hit: " << iHit << endl;
                    logFile << "Cryostat: " << cryostatIndex << "; TPC: " << tpcIndex << "; Plane: " << planeIndex << "; Wire: " << wireIndex << endl;
                    logFile << "Channel no.: " << channelNo << endl;
                    logFile << "ADC: " << adc << endl;
                    logFile << "TDC: " << tdc << endl;
                    logFile << "Origin: " << hitOrigin << endl;
                    logFile << "PDG: " << pdg << endl;
                    logFile << "Energy calculated over all contributions (MeV): " << totalHitEnergy << endl;
                    logFile << "Energy from the best track id (MeV): " << bestTrackIdEnergy << endl;
                    logFile << "Truth Energy (MeV): " << truthEnergy << endl << endl;
                }

            if(fDebug)
                {
                    ProtoDuneDL::AllHitsStruct tempAllHits;
                    tempAllHits.event_no = nEvents;
                    tempAllHits.event_id = eventId;
                    tempAllHits.adc = adc;
                    tempAllHits.tdc = tdc;
                    tempAllHits.channel_number = channelNo;
                    tempAllHits.tpc_index = tpcIndex;
                    tempAllHits.plane_index = planeIndex;
                    tempAllHits.wire_index = wireIndex;
                    tempAllHits.global_wire_index = globalWireIndex;
                    tempAllHits.global_plane_index = globalPlaneIndex;
                    tempAllHits.best_track_id = bestTrackId;
                    tempAllHits.pdg = pdg;
                    tempAllHits.total_hit_energy = totalHitEnergy;
                    tempAllHits.best_track_id_energy = bestTrackIdEnergy;
                    tempAllHits.true_energy = truthEnergy;
                    tempAllHits.origin = hitOrigin;
                    pAllHitsStruct->push_back(tempAllHits);
                }

            // Keep the hit only if it can be matched to the primary
            bool keepHit = true;
            if(isMC)
                {
                    keepHit = (mapPrimaryToHits.find(bestTrackId) != mapPrimaryToHits.end()) && (mapPrimaryToHits[bestTrackId] == hitOrigin);
                }
            if(keepHit)
                {
                    ProtoDuneDL::HitsStruct tempHits;
                    tempHits.adc = adc;
                    tempHits.tdc = tdc;
                    tempHits.tpc_index = tpcIndex;
                    tempHits.plane_index = planeIndex;
                    tempHits.wire_index = wireIndex;
                    tempHits.origin = hitOrigin;
                    pHitsStruct->push_back(tempHits);
                }
        }

    tHitsTree->Fill();

    if(fDebug)
        {
            tAllHitsTree->Fill();
        }
    logFile << endl;
}//analyze
}// namespace ProtoDuneDL

