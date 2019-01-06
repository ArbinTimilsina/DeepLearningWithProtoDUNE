#ifndef DATA_STRUCT_H
#define DATA_STRUCT_H

namespace ProtoDuneDL
{
    struct HitsStruct
    {
      	double adc;
	unsigned int tdc;
	unsigned int tpc_index;
        unsigned int plane_index;
        unsigned int wire_index;
	unsigned int origin;
    };

    struct AllHitsStruct
    {
	unsigned int event_no;
	unsigned int event_id;
	double adc;
	double tdc;
	unsigned int channel_number;
	unsigned int tpc_index;
	unsigned int plane_index;
	unsigned int wire_index;
	unsigned int global_wire_index;
	unsigned int global_plane_index;
	int best_track_id;
	int pdg;
	double total_hit_energy;
	double best_track_id_energy;
	double true_energy;
	unsigned int origin;
    };

    struct PrimaryParticleStruct
    {
	unsigned int event_no;
	unsigned int event_id;
	unsigned int origin;
	unsigned int n_particles;
	int pdg;
	double energy;
	int gen_track_id;
	int geant_track_id;
    };
}
#endif
