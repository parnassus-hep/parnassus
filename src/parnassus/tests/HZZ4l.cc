// pp -> H -> ZZ -> (l+l-)(l+l-), write HepMC3
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"

using namespace Pythia8;

int main() {
  Pythia pythia;

  // Random seed
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = 42");

  // Command file for rest of config
  pythia.readFile("HZZ4l.cmnd");

  if (!pythia.init()) return 1;

  // HepMC3 writer & converter.
  HepMC3::WriterAscii writer("HZZ4l_1k_benchmark.hepmc"); // or WriterGZ("events.hepmc.gz")
  HepMC3::Pythia8ToHepMC3 toHepMC;

  const int nEventsTarget = 1000;
  int nWritten = 0;

  // Generate exactly nEventsTarget successful events (retry on failed ones).
  for (int iEvent = 0; nWritten < nEventsTarget; /* no ++ here */) {
    if (!pythia.next()) continue; // event failed, try again

    HepMC3::GenEvent hepmcEvent(HepMC3::Units::GEV, HepMC3::Units::MM);
    toHepMC.fill_next_event(pythia, &hepmcEvent);
    writer.write_event(hepmcEvent);

    ++nWritten;
    ++iEvent;
  }

  pythia.stat();
  return 0;
}
