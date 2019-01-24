#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace ProtoDuneDL
{
    enum Labels
    {
	Background = 0,
	Beam = 1,
	NotBeam = 2,
    };

    const unsigned int WiresPerPlane[3] = {1148, 1148, 480}; // Plane: U, V, W
    const unsigned int WiresOffsetPerTPC[3] = {400, 400, 480};
    const unsigned int GlobalWiresPerPlane[3] = {1948, 1948, 1440}; // 2 * WiresOffsetPerTPC + WiresPerPlane
    const unsigned int TDCsPerPlane[3] = {6000, 6000, 6000};

    const unsigned int MaxPlanes = 3;
    const unsigned int MaxWires = 320;
    const unsigned int MaxTDCs = 320;
}
#endif
